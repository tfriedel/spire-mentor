using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace SpireMentor;

/// <summary>Starts the advisor without inheriting the game's open save files.
/// Process.Start on Windows enables unrestricted handle inheritance.</summary>
internal sealed class WindowsChildProcess : IDisposable
{
    public Process Process { get; }
    public StreamReader StandardOutput { get; }
    public StreamReader StandardError { get; }

    private WindowsChildProcess(Process process, Stream stdout, Stream stderr)
    {
        Process = process;
        // Match Process.StandardOutput/StandardError, including GUI processes
        // without a console (Python then writes using the Windows ANSI code page).
        int codePage = (int)GetConsoleCP();
        if (codePage == 0)
        {
            codePage = (int)GetACP();
        }
        Encoding encoding = CodePagesEncodingProvider.Instance.GetEncoding(codePage) ?? Encoding.GetEncoding(codePage);
        StandardOutput = new StreamReader(stdout, encoding);
        StandardError = new StreamReader(stderr, encoding);
    }

    public static WindowsChildProcess Start(string fileName, string arguments, string? workingDirectory,
        Action<Process>? beforeResume = null)
    {
        if (!OperatingSystem.IsWindows())
        {
            throw new PlatformNotSupportedException("The advisor launcher requires Windows.");
        }
        ArgumentException.ThrowIfNullOrWhiteSpace(fileName);

        AnonymousPipeServerStream? stdout = null;
        AnonymousPipeServerStream? stderr = null;
        Process? process = null;
        bool started = false;
        try
        {
            stdout = new AnonymousPipeServerStream(PipeDirection.In, HandleInheritability.Inheritable);
            stderr = new AnonymousPipeServerStream(PipeDirection.In, HandleInheritability.Inheritable);
            using var stdin = new AnonymousPipeServerStream(PipeDirection.Out, HandleInheritability.Inheritable);
            ProcessInformation info = CreateSuspended(fileName, arguments, workingDirectory,
                stdin.ClientSafePipeHandle.DangerousGetHandle(),
                stdout.ClientSafePipeHandle.DangerousGetHandle(),
                stderr.ClientSafePipeHandle.DangerousGetHandle());
            using var nativeProcess = new SafeProcessHandle(info.Process, ownsHandle: true);
            using var nativeThread = new SafeWaitHandle(info.Thread, ownsHandle: true);
            try
            {
                stdout.DisposeLocalCopyOfClientHandle();
                stderr.DisposeLocalCopyOfClientHandle();
                stdin.DisposeLocalCopyOfClientHandle();
                stdin.Dispose(); // The advisor never receives interactive stdin; expose EOF.

                process = Process.GetProcessById(info.ProcessId);
                _ = process.Handle; // Keep a handle even if the child exits immediately after resume.
                beforeResume?.Invoke(process);
                var child = new WindowsChildProcess(process, stdout, stderr);
                if (ResumeThread(nativeThread) == uint.MaxValue)
                {
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "Could not resume advisor.");
                }
                started = true;
                return child;
            }
            catch (Exception ex)
            {
                if (!TerminateProcess(nativeProcess, 1))
                {
                    throw new AggregateException("Advisor startup failed and cleanup could not terminate it.",
                        ex, new Win32Exception(Marshal.GetLastWin32Error()));
                }
                throw;
            }
        }
        finally
        {
            if (!started)
            {
                stdout?.Dispose();
                stderr?.Dispose();
                process?.Dispose();
            }
        }
    }

    private static ProcessInformation CreateSuspended(string fileName, string arguments,
        string? workingDirectory, IntPtr stdin, IntPtr stdout, IntPtr stderr)
    {
        IntPtr size = IntPtr.Zero;
        // The size query intentionally fails with ERROR_INSUFFICIENT_BUFFER.
        InitializeProcThreadAttributeList(IntPtr.Zero, 1, 0, ref size);
        if (size == IntPtr.Zero)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error(), "Could not size advisor handle allowlist.");
        }
        IntPtr attributes = Marshal.AllocHGlobal(size);
        IntPtr handles = IntPtr.Zero;
        bool initialized = false;
        try
        {
            initialized = InitializeProcThreadAttributeList(attributes, 1, 0, ref size);
            if (!initialized)
            {
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Could not initialize advisor handle allowlist.");
            }
            handles = Marshal.AllocHGlobal(3 * IntPtr.Size);
            Marshal.Copy(new[] { stdin, stdout, stderr }, 0, handles, 3);
            if (!UpdateProcThreadAttribute(attributes, 0, new IntPtr(0x20002),
                handles, new IntPtr(3 * IntPtr.Size), IntPtr.Zero, IntPtr.Zero))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Could not restrict advisor handle inheritance.");
            }
            var startup = new StartupInfoEx
            {
                StartupInfo = new StartupInfo
                {
                    Size = Marshal.SizeOf<StartupInfoEx>(),
                    Flags = 0x100, // STARTF_USESTDHANDLES
                    StandardInput = stdin,
                    StandardOutput = stdout,
                    StandardError = stderr,
                },
                AttributeList = attributes,
            };
            var commandLine = new StringBuilder($"\"{fileName}\" {arguments}");
            // CREATE_SUSPENDED | CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT.
            // Adopt into the kill-on-close job before any child code can run.
            if (!CreateProcess(null, commandLine, IntPtr.Zero, IntPtr.Zero, true,
                0x4 | 0x08000000 | 0x00080000, IntPtr.Zero,
                string.IsNullOrEmpty(workingDirectory) ? null : workingDirectory,
                ref startup, out ProcessInformation info))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error(), $"Could not start advisor '{fileName}'.");
            }
            return info;
        }
        finally
        {
            if (initialized)
            {
                DeleteProcThreadAttributeList(attributes);
            }
            Marshal.FreeHGlobal(attributes);
            Marshal.FreeHGlobal(handles);
        }
    }

    public void Dispose()
    {
        StandardOutput.Dispose();
        StandardError.Dispose();
        Process.Dispose();
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct StartupInfo
    {
        public int Size;
        public IntPtr Reserved, Desktop, Title;
        public int X, Y, XSize, YSize, XCountChars, YCountChars, FillAttribute, Flags;
        public short ShowWindow, ReservedSize;
        public IntPtr ReservedData, StandardInput, StandardOutput, StandardError;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct StartupInfoEx
    {
        public StartupInfo StartupInfo;
        public IntPtr AttributeList;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessInformation
    {
        public IntPtr Process, Thread;
        public int ProcessId, ThreadId;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool InitializeProcThreadAttributeList(IntPtr list, int count, int flags, ref IntPtr size);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool UpdateProcThreadAttribute(IntPtr list, uint flags, IntPtr attribute,
        IntPtr value, IntPtr size, IntPtr previousValue, IntPtr returnSize);

    [DllImport("kernel32.dll")]
    private static extern void DeleteProcThreadAttributeList(IntPtr list);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateProcess(string? applicationName, StringBuilder commandLine,
        IntPtr processAttributes, IntPtr threadAttributes, bool inheritHandles, uint creationFlags,
        IntPtr environment, string? currentDirectory, ref StartupInfoEx startupInfo, out ProcessInformation processInfo);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint ResumeThread(SafeWaitHandle thread);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateProcess(SafeProcessHandle process, uint exitCode);

    [DllImport("kernel32.dll")]
    private static extern uint GetConsoleCP();

    [DllImport("kernel32.dll")]
    private static extern uint GetACP();
}
