// RefCheck <SpireMentor.dll> <sts2.dll> [--assembly sts2] [--optional <file>]
//
// Lists every reference SpireMentor.dll makes into the game assembly that the
// given sts2.dll cannot satisfy: a missing type, or a method / field whose
// name + signature (return type, parameter types, generic arity) is not
// declared on that type or a base class inside sts2. These are exactly the
// references that compile against one game build and then throw
// MissingMethodException / MissingFieldException / TypeLoadException at JIT
// time on another -- e.g. a member whose type went `uint` -> `ulong` still
// compiles from source (`var`) but no longer binds in IL.
//
// Exit: 0 all resolve, 1 unresolved references (listed), 2 usage / IO error.
// Pure metadata: neither assembly is loaded or executed.
//
// Name-based binding (GH #807): a second pass checks the targets the mod names
// by STRING, which no member reference carries -- `[HarmonyPatch(typeof(T),
// "Name")]` (class- and method-level, `nameof` included) and the IL pattern
// `ldtoken T ... ldstr "name" ... call AccessTools.Method/Field/Property...`.
// Each must name a member of T (or one of T's sts2 base classes) by name; the
// signature is not checked, because the strings do not carry one. Reflection
// on a runtime type (`obj.GetType().GetMethod("x")`) has no static type and
// stays unchecked.

using System.Collections.Immutable;
using System.Reflection.Emit;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: RefCheck <SpireMentor.dll> <sts2.dll> [--assembly sts2] [--optional <file>]");
    return 2;
}
string gameAsm = "sts2";
// `ShortTypeName::name` targets a build may lack on purpose (tests/compile/RefCheck/
// optional-names.txt): `#` starts a comment.
var optionalNames = new HashSet<string>(StringComparer.Ordinal);
for (int i = 2; i < args.Length; i++)
{
    if (args[i] == "--assembly" && i + 1 < args.Length) gameAsm = args[++i];
    else if (args[i] == "--optional" && i + 1 < args.Length)
    {
        foreach (string line in File.ReadAllLines(args[++i]))
        {
            string entry = line.Split('#', 2)[0].Trim();
            if (entry.Length > 0) optionalNames.Add(entry);
        }
    }
}

using var modPe = new PEReader(File.OpenRead(args[0]));
using var gamePe = new PEReader(File.OpenRead(args[1]));
MetadataReader mod = modPe.GetMetadataReader();
MetadataReader game = gamePe.GetMetadataReader();
var names = new NameProvider();

// ---- index the game assembly ------------------------------------------------
var typeDefs = new Dictionary<string, TypeDefinitionHandle>();
foreach (TypeDefinitionHandle h in game.TypeDefinitions)
{
    typeDefs[NameProvider.FullName(game, h)] = h;
}

string? BaseTypeName(TypeDefinitionHandle h)
{
    EntityHandle b = game.GetTypeDefinition(h).BaseType;
    if (b.IsNil) return null;
    return b.Kind switch
    {
        HandleKind.TypeDefinition => NameProvider.FullName(game, (TypeDefinitionHandle)b),
        HandleKind.TypeReference => NameProvider.FullName(game, (TypeReferenceHandle)b),
        HandleKind.TypeSpecification => NameProvider.GenericRoot(
            game.GetTypeSpecification((TypeSpecificationHandle)b).DecodeSignature(names, null)),
        _ => null,
    };
}

// Searches `type` and its sts2 base classes. Reaching a base outside sts2
// (System.Object, a Godot node) is a miss: Roslyn emits a member reference's
// parent as the type that DECLARES the member, so a reference parented on an
// sts2 type names a member sts2 itself must declare.
string Resolve(string type, string member, string sig, bool isField)
{
    string? t = type;
    while (t != null)
    {
        if (!typeDefs.TryGetValue(t, out TypeDefinitionHandle h))
        {
            return t == type ? "type missing" : "not found";
        }
        TypeDefinition td = game.GetTypeDefinition(h);
        if (isField)
        {
            foreach (FieldDefinitionHandle fh in td.GetFields())
            {
                FieldDefinition f = game.GetFieldDefinition(fh);
                if (game.GetString(f.Name) == member && f.DecodeSignature(names, null) == sig) return "found";
            }
        }
        else
        {
            foreach (MethodDefinitionHandle mh in td.GetMethods())
            {
                MethodDefinition m = game.GetMethodDefinition(mh);
                if (game.GetString(m.Name) == member && NameProvider.Sig(m.DecodeSignature(names, null)) == sig) return "found";
            }
        }
        t = BaseTypeName(h);
    }
    return "not found";
}

string? Candidates(string type, string member, bool isField)
{
    if (!typeDefs.TryGetValue(type, out TypeDefinitionHandle h)) return null;
    TypeDefinition td = game.GetTypeDefinition(h);
    var found = new List<string>();
    if (isField)
    {
        foreach (FieldDefinitionHandle fh in td.GetFields())
        {
            FieldDefinition f = game.GetFieldDefinition(fh);
            if (game.GetString(f.Name) == member) found.Add(f.DecodeSignature(names, null));
        }
    }
    else
    {
        foreach (MethodDefinitionHandle mh in td.GetMethods())
        {
            MethodDefinition m = game.GetMethodDefinition(mh);
            if (game.GetString(m.Name) == member) found.Add(NameProvider.Sig(m.DecodeSignature(names, null)));
        }
    }
    return found.Count == 0 ? null : string.Join(" | ", found);
}

// ---- walk the mod's references -----------------------------------------------
bool IsGameRef(TypeReferenceHandle h)
{
    TypeReference tr = mod.GetTypeReference(h);
    EntityHandle scope = tr.ResolutionScope;
    while (scope.Kind == HandleKind.TypeReference)
    {
        scope = mod.GetTypeReference((TypeReferenceHandle)scope).ResolutionScope;
    }
    return scope.Kind == HandleKind.AssemblyReference
        && mod.GetString(mod.GetAssemblyReference((AssemblyReferenceHandle)scope).Name) == gameAsm;
}

var gameTypeRefs = new HashSet<string>();
var problems = new SortedSet<string>(StringComparer.Ordinal);
int checkedTypes = 0, checkedMembers = 0;
foreach (TypeReferenceHandle h in mod.TypeReferences)
{
    if (!IsGameRef(h)) continue;
    string n = NameProvider.FullName(mod, h);
    gameTypeRefs.Add(n);
    checkedTypes++;
    if (!typeDefs.ContainsKey(n)) problems.Add($"TYPE   {n}: missing");
}

foreach (MemberReferenceHandle h in mod.MemberReferences)
{
    MemberReference mr = mod.GetMemberReference(h);
    string? parent = mr.Parent.Kind switch
    {
        HandleKind.TypeReference when IsGameRef((TypeReferenceHandle)mr.Parent)
            => NameProvider.FullName(mod, (TypeReferenceHandle)mr.Parent),
        HandleKind.TypeSpecification => NameProvider.GenericRoot(
            mod.GetTypeSpecification((TypeSpecificationHandle)mr.Parent).DecodeSignature(names, null)),
        _ => null,
    };
    if (parent == null || !gameTypeRefs.Contains(parent)) continue;
    string member = mod.GetString(mr.Name);
    bool isField = mr.GetKind() == MemberReferenceKind.Field;
    string sig = isField ? mr.DecodeFieldSignature(names, null) : NameProvider.Sig(mr.DecodeMethodSignature(names, null));
    checkedMembers++;
    string why = Resolve(parent, member, sig, isField);
    if (why != "found")
    {
        string? have = why == "not found" ? Candidates(parent, member, isField) : null;
        problems.Add($"{(isField ? "FIELD " : "METHOD")} {parent}::{member} {sig}: {why}"
            + (have != null ? $" (game has: {have})" : ""));
    }
}

// ---- name-based targets (Harmony attributes, AccessTools by name) -------------
// Does `type` (or an sts2 base) declare a member called `name`? `kind` is
// "method", "field" or "property" (a property matches its get_/set_ accessor).
bool HasNamed(string type, string name, string kind)
{
    string? t = type;
    while (t != null && typeDefs.TryGetValue(t, out TypeDefinitionHandle h))
    {
        TypeDefinition td = game.GetTypeDefinition(h);
        if (kind == "field")
        {
            foreach (FieldDefinitionHandle fh in td.GetFields())
                if (game.GetString(game.GetFieldDefinition(fh).Name) == name) return true;
        }
        else
        {
            foreach (MethodDefinitionHandle mh in td.GetMethods())
            {
                string m = game.GetString(game.GetMethodDefinition(mh).Name);
                if (kind == "method" ? m == name : m == "get_" + name || m == "set_" + name) return true;
            }
        }
        t = BaseTypeName(h);
    }
    return false;
}

int checkedNamed = 0;
var optionalSeen = new HashSet<string>();
void CheckNamed(string origin, string type, string name, string kind)
{
    // Keyed by the type's name without its namespace, so the list carries no
    // game namespaces into the public export (its content-marker ratchet).
    string key = type[(type.LastIndexOf('.') + 1)..] + "::" + name;
    if (optionalNames.Contains(key))
    {
        optionalSeen.Add(key);
        if (!typeDefs.ContainsKey(type) || !HasNamed(type, name, kind)) return;
    }
    if (!typeDefs.ContainsKey(type))
    {
        // A game type the mod names that this build lacks is reported by the
        // type-reference pass above; a non-game type is not ours to check.
        return;
    }
    checkedNamed++;
    if (!HasNamed(type, name, kind))
    {
        problems.Add($"NAMED  {type}::{name} ({kind}, {origin}): not found");
    }
}

// The game type an `ldtoken` operand / attribute type argument names, or null.
string? GameTypeOf(EntityHandle h) => h.Kind switch
{
    HandleKind.TypeReference when IsGameRef((TypeReferenceHandle)h) => NameProvider.FullName(mod, (TypeReferenceHandle)h),
    HandleKind.TypeSpecification => NameProvider.GenericRoot(
        mod.GetTypeSpecification((TypeSpecificationHandle)h).DecodeSignature(names, null)) is var g && gameTypeRefs.Contains(g) ? g : null,
    _ => null,
};

string? AttributeTypeName(CustomAttributeHandle ah)
{
    CustomAttribute ca = mod.GetCustomAttribute(ah);
    EntityHandle parent = ca.Constructor.Kind == HandleKind.MemberReference
        ? mod.GetMemberReference((MemberReferenceHandle)ca.Constructor).Parent
        : default;
    return parent.Kind == HandleKind.TypeReference ? NameProvider.FullName(mod, (TypeReferenceHandle)parent) : null;
}

// A serialized `typeof` argument ("Ns.Outer+Inner, sts2, Version=...") as the
// FullName spelling, or null when it names another assembly.
string? SerializedGameType(string serialized)
{
    string[] parts = serialized.Split(',', 3);
    if (parts.Length < 2 || parts[1].Trim() != gameAsm) return null;
    return parts[0].Trim().Replace('+', '/');
}

// Harmony's MethodType: Normal 0, Getter 1, Setter 2, Constructor 3.
(string? Type, string? Name, int MethodType) HarmonyArgs(CustomAttributeHandle ah)
{
    var value = mod.GetCustomAttribute(ah).DecodeValue(new AttrTypeProvider());
    string? type = null, name = null;
    int methodType = 0;
    foreach (var a in value.FixedArguments)
    {
        switch (a.Value)
        {
            case string s when a.Type == "System.Type": type ??= SerializedGameType(s); break;
            case string s: name ??= s; break;
            case int i when a.Type == "HarmonyLib.MethodType": methodType = i; break;
        }
    }
    return (type, name, methodType);
}

void CheckHarmonyTarget(string origin, string? type, string? name, int methodType)
{
    if (type == null) return;
    switch (methodType)
    {
        case 0 when name != null: CheckNamed(origin, type, name, "method"); break;
        case 1 or 2 when name != null: CheckNamed(origin, type, (methodType == 1 ? "get_" : "set_") + name, "method"); break;
        case 3: CheckNamed(origin, type, ".ctor", "method"); break;
    }
}

foreach (TypeDefinitionHandle th in mod.TypeDefinitions)
{
    TypeDefinition td = mod.GetTypeDefinition(th);
    string owner = NameProvider.FullName(mod, th);
    (string? Type, string? Name, int MethodType) cls = (null, null, 0);
    bool classPatched = false;
    foreach (CustomAttributeHandle ah in td.GetCustomAttributes())
    {
        if (AttributeTypeName(ah) != "HarmonyLib.HarmonyPatch") continue;
        var a = HarmonyArgs(ah);
        classPatched = true;
        cls = (a.Type ?? cls.Type, a.Name ?? cls.Name, a.MethodType != 0 ? a.MethodType : cls.MethodType);
    }
    bool methodNamed = false;
    foreach (MethodDefinitionHandle mh in td.GetMethods())
    {
        foreach (CustomAttributeHandle ah in mod.GetMethodDefinition(mh).GetCustomAttributes())
        {
            if (AttributeTypeName(ah) != "HarmonyLib.HarmonyPatch") continue;
            var a = HarmonyArgs(ah);
            methodNamed = true;
            CheckHarmonyTarget(owner + "." + mod.GetString(mod.GetMethodDefinition(mh).Name) + " [HarmonyPatch]",
                a.Type ?? cls.Type, a.Name ?? cls.Name, a.MethodType != 0 ? a.MethodType : cls.MethodType);
        }
    }
    if (classPatched && !methodNamed)
    {
        CheckHarmonyTarget(owner + " [HarmonyPatch]", cls.Type, cls.Name, cls.MethodType);
    }
}

// IL: `ldtoken T; call GetTypeFromHandle; ldstr "name"` feeding an AccessTools
// lookup within a few instructions. Operand sizes come from the BCL's own
// opcode table.
var operandTypes = new Dictionary<short, OperandType>();
foreach (var f in typeof(OpCodes).GetFields())
{
    if (f.GetValue(null) is OpCode op) operandTypes[op.Value] = op.OperandType;
}
var accessKinds = new Dictionary<string, string>
{
    ["Method"] = "method", ["DeclaredMethod"] = "method",
    ["Field"] = "field", ["DeclaredField"] = "field",
    ["Property"] = "property", ["DeclaredProperty"] = "property",
    ["PropertyGetter"] = "property", ["PropertySetter"] = "property",
    ["DeclaredPropertyGetter"] = "property", ["DeclaredPropertySetter"] = "property",
};
foreach (MethodDefinitionHandle mh in mod.MethodDefinitions)
{
    MethodDefinition md = mod.GetMethodDefinition(mh);
    if (md.RelativeVirtualAddress == 0) continue;
    BlobReader il = modPe.GetMethodBody(md.RelativeVirtualAddress).GetILReader();
    string where = NameProvider.FullName(mod, md.GetDeclaringType()) + "." + mod.GetString(md.Name);
    string? lastTok = null; int lastTokAt = -100;
    (string Type, string Name, int At)? pair = null;
    int n = 0;
    while (il.RemainingBytes > 0)
    {
        n++;
        short code = il.ReadByte();
        if (code == 0xFE) code = unchecked((short)(0xFE00 | il.ReadByte()));
        OperandType ot = operandTypes.TryGetValue(code, out OperandType o) ? o : OperandType.InlineNone;
        if (code == (short)ILOpCode.Ldtoken)
        {
            lastTok = GameTypeOf(MetadataTokens.EntityHandle(il.ReadInt32()));
            lastTokAt = n;
            continue;
        }
        if (code == (short)ILOpCode.Ldstr)
        {
            string s = mod.GetUserString(MetadataTokens.UserStringHandle(il.ReadInt32()));
            pair = lastTok != null && n - lastTokAt <= 2 ? (lastTok, s, n) : null;
            continue;
        }
        if (code is (short)ILOpCode.Call or (short)ILOpCode.Callvirt)
        {
            EntityHandle target = MetadataTokens.EntityHandle(il.ReadInt32());
            if (pair is { } p && n - p.At <= 6 && target.Kind == HandleKind.MemberReference)
            {
                MemberReference mr = mod.GetMemberReference((MemberReferenceHandle)target);
                if (mr.Parent.Kind == HandleKind.TypeReference
                    && NameProvider.FullName(mod, (TypeReferenceHandle)mr.Parent) == "HarmonyLib.AccessTools"
                    && accessKinds.TryGetValue(mod.GetString(mr.Name), out string? kind))
                {
                    CheckNamed(where + " AccessTools." + mod.GetString(mr.Name), p.Type, p.Name, kind);
                    pair = null;
                }
            }
            continue;
        }
        switch (ot)
        {
            case OperandType.InlineNone: break;
            case OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar: il.Offset += 1; break;
            case OperandType.InlineVar: il.Offset += 2; break;
            case OperandType.InlineI8 or OperandType.InlineR: il.Offset += 8; break;
            case OperandType.InlineSwitch:
                // Read the target count FIRST: `il.Offset += 4 * il.ReadInt32()`
                // reads Offset before ReadInt32 advances past the count, lands
                // 4 bytes short and walks the jump table as opcodes -- silently
                // skipping later references, or running off the method body.
                int targets = il.ReadInt32();
                il.Offset += 4 * targets;
                break;
            default: il.Offset += 4; break;
        }
    }
}

foreach (string stale in optionalNames.Where(o => !optionalSeen.Contains(o)))
{
    problems.Add($"OPTIONAL {stale}: listed in --optional but the mod no longer looks it up; remove the entry");
}

Console.WriteLine($"RefCheck: {checkedTypes} {gameAsm} type refs, {checkedMembers} member refs, {checkedNamed} name-based targets checked against {args[1]}");
foreach (string p in problems) Console.WriteLine("  UNRESOLVED " + p);
Console.WriteLine(problems.Count == 0 ? "RefCheck: OK, every reference binds" : $"RefCheck: {problems.Count} unresolved reference(s)");
return problems.Count == 0 ? 0 : 1;

/// <summary>Decodes custom-attribute blobs: a Type argument's value is its
/// serialized name, typed "System.Type"; an enum is typed by its name (its
/// underlying type is assumed int32, true of HarmonyLib's).</summary>
sealed class AttrTypeProvider : ICustomAttributeTypeProvider<string>
{
    public string GetPrimitiveType(PrimitiveTypeCode typeCode) => typeCode.ToString();
    public string GetSystemType() => "System.Type";
    public string GetSZArrayType(string elementType) => elementType + "[]";
    public string GetTypeFromDefinition(MetadataReader reader, TypeDefinitionHandle handle, byte rawTypeKind) => NameProvider.FullName(reader, handle);
    public string GetTypeFromReference(MetadataReader reader, TypeReferenceHandle handle, byte rawTypeKind) => NameProvider.FullName(reader, handle);
    public string GetTypeFromSerializedName(string name) => name;
    public PrimitiveTypeCode GetUnderlyingEnumType(string type) => PrimitiveTypeCode.Int32;
    public bool IsSystemType(string type) => type == "System.Type";
}
sealed class NameProvider : ISignatureTypeProvider<string, object?>
{
    public static string FullName(MetadataReader r, TypeDefinitionHandle h)
    {
        TypeDefinition td = r.GetTypeDefinition(h);
        string name = r.GetString(td.Name);
        TypeDefinitionHandle decl = td.GetDeclaringType();
        if (!decl.IsNil) return FullName(r, decl) + "/" + name;
        string ns = r.GetString(td.Namespace);
        return ns.Length == 0 ? name : ns + "." + name;
    }

    public static string FullName(MetadataReader r, TypeReferenceHandle h)
    {
        TypeReference tr = r.GetTypeReference(h);
        string name = r.GetString(tr.Name);
        if (tr.ResolutionScope.Kind == HandleKind.TypeReference)
        {
            return FullName(r, (TypeReferenceHandle)tr.ResolutionScope) + "/" + name;
        }
        string ns = r.GetString(tr.Namespace);
        return ns.Length == 0 ? name : ns + "." + name;
    }

    public static string GenericRoot(string s)
    {
        int i = s.IndexOf('<');
        return i < 0 ? s : s[..i];
    }

    public static string Sig(MethodSignature<string> m)
        => $"{m.ReturnType}({string.Join(",", m.ParameterTypes)})`{m.GenericParameterCount}{(m.Header.IsInstance ? "" : " static")}";

    public string GetPrimitiveType(PrimitiveTypeCode typeCode) => typeCode.ToString();
    public string GetTypeFromDefinition(MetadataReader reader, TypeDefinitionHandle handle, byte rawTypeKind) => FullName(reader, handle);
    public string GetTypeFromReference(MetadataReader reader, TypeReferenceHandle handle, byte rawTypeKind) => FullName(reader, handle);
    public string GetTypeFromSpecification(MetadataReader reader, object? genericContext, TypeSpecificationHandle handle, byte rawTypeKind)
        => reader.GetTypeSpecification(handle).DecodeSignature(this, genericContext);
    public string GetSZArrayType(string elementType) => elementType + "[]";
    public string GetArrayType(string elementType, ArrayShape shape) => elementType + "[" + new string(',', shape.Rank - 1) + "]";
    public string GetByReferenceType(string elementType) => elementType + "&";
    public string GetPointerType(string elementType) => elementType + "*";
    public string GetPinnedType(string elementType) => elementType;
    public string GetGenericInstantiation(string genericType, ImmutableArray<string> typeArguments)
        => genericType + "<" + string.Join(",", typeArguments) + ">";
    public string GetGenericMethodParameter(object? genericContext, int index) => "!!" + index;
    public string GetGenericTypeParameter(object? genericContext, int index) => "!" + index;
    public string GetFunctionPointerType(MethodSignature<string> signature) => "fnptr(" + Sig(signature) + ")";
    public string GetModifiedType(string modifier, string unmodifiedType, bool isRequired)
        => isRequired ? unmodifiedType + " modreq(" + modifier + ")" : unmodifiedType;
}
