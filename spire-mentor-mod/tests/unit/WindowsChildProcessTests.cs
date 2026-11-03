using System.ComponentModel;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using SpireMentor;

internal static class WindowsChildProcessTests
{
    public static int RunChild(string releaseFile, int? codePage)
    {
        if (codePage is { } encodingCodePage)
        {
            Console.OutputEncoding = System.Text.CodePagesEncodingProvider.Instance.GetEncoding(encodingCodePage)
                ?? System.Text.Encoding.GetEncoding(encodingCodePage);
        }
        Console.WriteLine("ready");
        Console.Error.WriteLine("stderr ready");
        DateTime deadline = DateTime.UtcNow.AddSeconds(30);
        while (!File.Exists(releaseFile))
        {
            if (DateTime.UtcNow >= deadline)
            {
                Console.Error.WriteLine("Timed out waiting for the parent test.");
                return 99;
            }
            Thread.Sleep(20);
        }
        Console.WriteLine("caf\u00e9");
        Console.WriteLine(Environment.CurrentDirectory);
        return 7;
    }

    public static void Run(Action<string, string?, string?> check)
    {
        string directory = Path.Combine(Path.GetTempPath(), $"advisor handles {Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string source = Path.Combine(directory, "current_run.save");
        string destination = Path.Combine(directory, "renamed.save");
        string release = Path.Combine(directory, "release child");
        string executable = Environment.ProcessPath ?? throw new InvalidOperationException("No test executable.");
        string arguments = $"--advisor-child \"{release}\"";
        if (Path.GetFileNameWithoutExtension(executable).Equals("dotnet", StringComparison.OrdinalIgnoreCase))
        {
            arguments = $"\"{Assembly.GetExecutingAssembly().Location}\" {arguments}";
        }
        try
        {
            int logCodePage;
            // Prove the fixture reproduces the game's lock with the old launcher.
            using (var file = OpenInheritableSave(source))
            using (var baseline = Process.Start(new ProcessStartInfo(executable, arguments)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                WorkingDirectory = directory,
            }) ?? throw new InvalidOperationException("Could not start baseline child."))
            {
                try
                {
                    check("baseline child ready", ReadLine(baseline.StandardOutput), "ready");
                    logCodePage = baseline.StandardOutput.CurrentEncoding.CodePage;
                    file.Dispose();
                    bool locked = false;
                    try
                    {
                        File.Move(source, destination);
                    }
                    catch (IOException ex) when ((ex.HResult & 0xffff) == 32)
                    {
                        locked = true;
                    }
                    check("baseline inherits save lock", locked.ToString(), "True");
                }
                finally
                {
                    StopChild(baseline, release);
                }
            }
            File.Delete(release);
            arguments += " " + logCodePage.ToString(System.Globalization.CultureInfo.InvariantCulture);

            using (var file = OpenInheritableSave(source))
            using (var child = WindowsChildProcess.Start(executable, arguments, directory))
            {
                try
                {
                    check("restricted child stdout", ReadLine(child.StandardOutput), "ready");
                    check("restricted child stderr", ReadLine(child.StandardError), "stderr ready");
                    file.Dispose();
                    File.Move(source, destination, overwrite: true);
                    File.Delete(destination);
                    check("save can be renamed and deleted while child is alive", child.Process.HasExited.ToString(), "False");
                    File.WriteAllText(release, "");
                    check("non-ASCII output matches Process encoding", ReadLine(child.StandardOutput), "caf\u00e9");
                    check("working directory preserved", ReadLine(child.StandardOutput), directory);
                    if (!child.Process.WaitForExit(10000))
                    {
                        throw new TimeoutException("Restricted child did not exit.");
                    }
                    check("child exit code preserved", child.Process.ExitCode.ToString(), "7");
                    check("stdout reaches EOF", ReadLine(child.StandardOutput), null);
                }
                finally
                {
                    StopChild(child.Process, release);
                }
            }

            bool failed = false;
            try
            {
                using var child = WindowsChildProcess.Start(Path.Combine(directory, "missing.exe"), "", directory);
            }
            catch (Win32Exception ex) when (ex.NativeErrorCode == 2)
            {
                failed = true;
            }
            check("launch failure is surfaced", failed.ToString(), "True");

            int failedChildId = 0;
            bool callbackFailed = false;
            try
            {
                using var child = WindowsChildProcess.Start(executable, arguments, directory, process =>
                {
                    failedChildId = process.Id;
                    throw new InvalidOperationException("simulated job setup failure");
                });
            }
            catch (InvalidOperationException ex) when (ex.Message == "simulated job setup failure")
            {
                callbackFailed = true;
            }
            check("pre-resume failure is surfaced", callbackFailed.ToString(), "True");
            try
            {
                using var failedChild = Process.GetProcessById(failedChildId);
                check("failed startup terminates suspended child", failedChild.WaitForExit(10000).ToString(), "True");
            }
            catch (ArgumentException)
            {
                check("failed startup terminates suspended child", "True", "True");
            }
        }
        finally
        {
            foreach (string path in new[] { source, destination, release })
            {
                File.Delete(path);
            }
            Directory.Delete(directory);
        }
    }

    private static FileStream OpenInheritableSave(string path)
    {
        var file = new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.ReadWrite);
        if (!SetHandleInformation(file.SafeFileHandle, 1, 1))
        {
            int error = Marshal.GetLastWin32Error();
            file.Dispose();
            throw new Win32Exception(error);
        }
        return file;
    }

    private static string? ReadLine(StreamReader reader) =>
        reader.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(10)).GetAwaiter().GetResult();

    private static void StopChild(Process process, string release)
    {
        File.WriteAllText(release, "");
        if (!process.WaitForExit(10000))
        {
            process.Kill(entireProcessTree: true);
            process.WaitForExit();
            throw new TimeoutException("Test child required forced cleanup.");
        }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetHandleInformation(Microsoft.Win32.SafeHandles.SafeFileHandle handle, uint mask, uint flags);
}
