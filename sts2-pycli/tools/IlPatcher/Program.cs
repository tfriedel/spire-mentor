using System;
using System.IO;
using Mono.Cecil;
using Mono.Cecil.Cil;

var dllPath = args[0];
Console.WriteLine($"Patching {dllPath}...");

var resolver = new DefaultAssemblyResolver();
var libDir = Path.GetDirectoryName(dllPath)!;
resolver.AddSearchDirectory(libDir);
// Also search for GodotSharp.dll in the GodotStubs output (fallback)
var stubsDir = Path.Combine(Path.GetDirectoryName(libDir)!, "src", "GodotStubs", "bin", "Debug", "net9.0");
if (Directory.Exists(stubsDir)) resolver.AddSearchDirectory(stubsDir);
var module = ModuleDefinition.ReadModule(dllPath, new ReaderParameters {
    AssemblyResolver = resolver,
    ReadingMode = ReadingMode.Deferred  // Don't force-resolve all references upfront
});

int patches = 0;

// Patch 1: Task.Yield() — make YieldAwaitable.YieldAwaiter.IsCompleted return true
// This prevents async deadlocks in headless mode
foreach (var type in module.Types)
{
    foreach (var nested in type.NestedTypes)
    {
        foreach (var nested2 in nested.NestedTypes)
        {
            if (nested2.Name.Contains("YieldAwaiter") || nested2.Name == "<>c")
            {
                foreach (var method in nested2.Methods)
                {
                    if (method.Name == "get_IsCompleted" && method.Body != null)
                    {
                        var il = method.Body.GetILProcessor();
                        il.Body.Instructions.Clear();
                        il.Emit(OpCodes.Ldc_I4_1);
                        il.Emit(OpCodes.Ret);
                        patches++;
                        Console.WriteLine($"  Patched {type.Name}.{nested.Name}.{nested2.Name}.IsCompleted");
                    }
                }
            }
        }
    }
}

// Patch 2: WaitUntilQueueIsEmptyOrWaitingOnNonPlayerDrivenAction → return Task.CompletedTask
foreach (var type in module.Types)
{
    foreach (var method in type.Methods)
    {
        if (method.Name == "WaitUntilQueueIsEmptyOrWaitingOnNonPlayerDrivenAction" && method.Body != null)
        {
            var il = method.Body.GetILProcessor();
            il.Body.Instructions.Clear();
            // return Task.CompletedTask
            var taskType = module.ImportReference(typeof(System.Threading.Tasks.Task));
            var completedProp = module.ImportReference(
                typeof(System.Threading.Tasks.Task).GetProperty("CompletedTask")!.GetGetMethod()!);
            il.Emit(OpCodes.Call, completedProp);
            il.Emit(OpCodes.Ret);
            patches++;
            Console.WriteLine($"  Patched {type.Name}.{method.Name} → Task.CompletedTask");
        }
    }
}

Console.WriteLine($"Applied {patches} patches");
var outPath = dllPath + ".patched";
module.Write(outPath);
module.Dispose();
File.Delete(dllPath);
File.Move(outPath, dllPath);
Console.WriteLine("Done!");
