// SimAiPartnerTransport.cs -- the in-memory network between the host and its
// AI partners (docs/plans/2026-11-05-ai-coop-partners.md, phase 1).
//
// A NetHost / NetClient pair the game's own net services run on, as they run
// on ENet or Steam: no socket, no port, no firewall prompt. Every packet is
// copied into the receiver's queue and delivered on the receiver's Update(),
// never inside the sender's call, so ordering and re-entrancy are those of a
// real network (one reliable, ordered channel per direction).
//
// Both base classes and both handler interfaces are the same in v0.107.1 and
// v0.111.0, except that v0.113.0 added a `string? debugReason = null` to the
// abstract NetHost.DisconnectClient. A class can only override the signature
// it was compiled against, and a concrete class that leaves the other
// build's abstract slot empty fails to LOAD (TypeLoadException), so the two
// hosts are abstract here and SimNetHosts emits the concrete subclass at run
// time against whichever DisconnectClient the game declares. What differs is wired in by
// reflection: NetHostGameService's private _netHost field (v0.111.0 has
// StartTestHost, v0.107.1 has nothing), and the connection initializer
// JoinFlow takes (`Connect(NetClientGameService, ...)` in v0.107.1,
// `Connect(INetClientGameService, ...)` in v0.111.0), which a DispatchProxy
// implements at run time against whichever interface the game has.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Connection;
using MegaCrit.Sts2.Core.Multiplayer.Transport;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Platform.Steam;

namespace SpireMentor;

internal readonly record struct SimLoopbackPacket(byte[] Bytes, NetTransferMode Mode, int Channel);

/// <summary>The host end. Its handler is the host's NetHostGameService.
/// Abstract: build one with <see cref="SimNetHosts.Create{T}"/>.</summary>
public abstract class SimLoopbackHost : NetHost
{
    /// <summary>The host's net id in a session of its own, as the game's own
    /// fastmp host uses (the null platform names it; SimAiPartnerNames shows
    /// the player's own name instead).</summary>
    public const ulong HostId = 1uL;

    private readonly ulong _hostId;

    private readonly Dictionary<ulong, SimLoopbackClient> _clients = new();
    private readonly Queue<ulong> _connected = new();
    private readonly Queue<(ulong Peer, NetErrorInfo Info)> _disconnected = new();
    private readonly Queue<(ulong From, SimLoopbackPacket Packet)> _inbox = new();
    private bool _running;

    /// <param name="hostId">The host's net id: <see cref="HostId"/> on its
    /// own, the game's host's (a Steam id) inside a <see cref="SimMixedHost"/>.</param>
    protected SimLoopbackHost(INetHostHandler handler, ulong hostId)
        : base(handler)
    {
        _hostId = hostId;
    }

    public override IEnumerable<ulong> ConnectedPeerIds => _clients.Keys.ToList();

    public override bool IsConnected => _running;

    public override ulong NetId => _hostId;

    public bool HasPeer(ulong peer) => _clients.ContainsKey(peer);

    public void Start() => _running = true;

    public override void Update()
    {
        // Connects before packets: a client's first packet (the v0.111.0
        // handshake) needs the host to know the peer.
        while (_connected.Count > 0)
        {
            _handler.OnPeerConnected(_connected.Dequeue());
        }
        while (_inbox.Count > 0)
        {
            (ulong from, SimLoopbackPacket p) = _inbox.Dequeue();
            if (_clients.ContainsKey(from))
            {
                _handler.OnPacketReceived(from, p.Bytes, p.Mode, p.Channel);
            }
        }
        while (_disconnected.Count > 0)
        {
            (ulong peer, NetErrorInfo info) = _disconnected.Dequeue();
            _handler.OnPeerDisconnected(peer, info);
        }
    }

    public override void SetHostIsClosed(bool closed)
    {
    }

    public override void SendMessageToClient(ulong peer, byte[] data, int count, NetTransferMode transfer, int lane = 0)
    {
        if (_clients.TryGetValue(peer, out SimLoopbackClient? client))
        {
            client.Deliver(new SimLoopbackPacket(Copy(data, count), transfer, lane));
        }
    }

    public override void SendMessageToAll(byte[] data, int count, NetTransferMode transfer, int lane = 0)
    {
        foreach (SimLoopbackClient client in _clients.Values.ToList())
        {
            client.Deliver(new SimLoopbackPacket(Copy(data, count), transfer, lane));
        }
    }

    /// <summary><c>DisconnectClient</c> of every build; see <see cref="SimNetHosts"/>.</summary>
    public void DisconnectClientCore(ulong peer, NetError why, bool immediately)
    {
        if (!_clients.Remove(peer, out SimLoopbackClient? client))
        {
            return;
        }
        client.HostDropped(ErrorInfo(why, ours: false));
        _disconnected.Enqueue((peer, ErrorInfo(why, ours: true)));
    }

    public override void StopHost(NetError why, bool immediately = false)
    {
        if (Stop(why))
        {
            _handler.OnDisconnected(ErrorInfo(why, ours: true));
        }
    }

    /// <summary>Drops every partner and stops, without telling the host's
    /// service: inside a <see cref="SimMixedHost"/> the game's own host
    /// reports the stop. False if already stopped.</summary>
    internal bool Stop(NetError why)
    {
        if (!_running)
        {
            return false;
        }
        foreach (ulong id in _clients.Keys.ToList())
        {
            DisconnectClientCore(id, why, immediately: false);
        }
        _running = false;
        _connected.Clear();
        _inbox.Clear();
        _disconnected.Clear();
        return true;
    }

    /// <summary>A disconnect's error; <paramref name="ours"/> = this end
    /// initiated it.</summary>
    internal static NetErrorInfo ErrorInfo(NetError error, bool ours) => new(error, ours);

    public override string? GetRawLobbyIdentifier() => null;

    internal bool Accept(SimLoopbackClient client)
    {
        if (!_running || _clients.ContainsKey(client.NetId))
        {
            return false;
        }
        _clients[client.NetId] = client;
        _connected.Enqueue(client.NetId);
        return true;
    }

    internal void FromClient(ulong from, SimLoopbackPacket packet) => _inbox.Enqueue((from, packet));

    internal void ClientLeft(ulong peerId, NetError reason)
    {
        if (_clients.Remove(peerId))
        {
            _disconnected.Enqueue((peerId, ErrorInfo(reason, ours: false)));
        }
    }

    internal static byte[] Copy(byte[] bytes, int length)
    {
        // The game's PacketWriter reuses its buffer for the next message.
        var copy = new byte[length];
        Buffer.BlockCopy(bytes, 0, copy, 0, length);
        return copy;
    }
}

/// <summary>
/// The game's own host (Steam, or ENet for fastmp) with AI partners next to
/// its human clients (plan, phase 4). The host's NetHostGameService holds
/// one NetHost; this one routes by peer id: a partner's packets to the
/// loopback, everyone else's to the game's host. Both ends already call the
/// same handler (the service), so receiving needs no routing. The platform
/// stays the game's: names and invites are Steam's as before.
/// </summary>
public abstract class SimMixedHost : NetHost
{
    private readonly NetHost _game;

    protected SimMixedHost(INetHostHandler handler, NetHost game, SimLoopbackHost partners)
        : base(handler)
    {
        _game = game;
        Partners = partners;
    }

    public SimLoopbackHost Partners { get; }

    public override IEnumerable<ulong> ConnectedPeerIds => _game.ConnectedPeerIds.Concat(Partners.ConnectedPeerIds).ToList();

    public override bool IsConnected => _game.IsConnected;

    public override ulong NetId => _game.NetId;

    public override void Update()
    {
        _game.Update();
        Partners.Update();
    }

    public override void SetHostIsClosed(bool closed) => _game.SetHostIsClosed(closed);

    public override void SendMessageToClient(ulong peer, byte[] data, int count, NetTransferMode transfer, int lane = 0)
    {
        if (Partners.HasPeer(peer))
        {
            Partners.SendMessageToClient(peer, data, count, transfer, lane);
        }
        else
        {
            _game.SendMessageToClient(peer, data, count, transfer, lane);
        }
    }

    public override void SendMessageToAll(byte[] data, int count, NetTransferMode transfer, int lane = 0)
    {
        _game.SendMessageToAll(data, count, transfer, lane);
        Partners.SendMessageToAll(data, count, transfer, lane);
    }

    /// <summary><c>DisconnectClient</c> of every build; see <see cref="SimNetHosts"/>.</summary>
    public void DisconnectClientCore(ulong peer, NetError why, bool immediately)
    {
        if (Partners.HasPeer(peer))
        {
            Partners.DisconnectClientCore(peer, why, immediately);
        }
        else
        {
            SimNetHosts.DisconnectClient(_game, peer, why, immediately);
        }
    }

    public override void StopHost(NetError why, bool immediately = false)
    {
        Partners.Stop(why);
        // The game's host reports the stop to the service, once.
        _game.StopHost(why, immediately);
    }

    public override string? GetRawLobbyIdentifier() => _game.GetRawLobbyIdentifier();
}

/// <summary>A partner's end. Its handler is the partner's NetClientGameService.</summary>
internal sealed class SimLoopbackClient : NetClient
{
    private readonly SimLoopbackHost _host;
    private readonly ulong _seatId;
    private readonly Queue<SimLoopbackPacket> _inbox = new();
    private NetErrorInfo? _dropped;
    private bool _connected;

    public SimLoopbackClient(INetClientHandler handler, SimLoopbackHost host, ulong seatId)
        : base(handler)
    {
        _host = host;
        _seatId = seatId;
    }

    public override bool IsConnected => _connected;

    public override ulong NetId => _seatId;

    public override ulong HostNetId => _host.NetId;

    /// <summary>Joins the host; the service then runs its side of the
    /// connection (v0.111.0: the handshake) from OnConnectedToHost.</summary>
    public NetErrorInfo? Connect()
    {
        _connected = true;
        if (!_host.Accept(this))
        {
            _connected = false;
            return SimLoopbackHost.ErrorInfo(NetError.InternalError, ours: false);
        }
        _handler.OnConnectedToHost();
        return null;
    }

    public override void Update()
    {
        while (_inbox.Count > 0)
        {
            SimLoopbackPacket p = _inbox.Dequeue();
            _handler.OnPacketReceived(_host.NetId, p.Bytes, p.Mode, p.Channel);
        }
        if (_dropped is NetErrorInfo info)
        {
            _dropped = null;
            _handler.OnDisconnectedFromHost(_host.NetId, info);
        }
    }

    public override void SendMessageToHost(byte[] data, int count, NetTransferMode transfer, int lane = 0)
    {
        if (_connected)
        {
            _host.FromClient(_seatId, new SimLoopbackPacket(SimLoopbackHost.Copy(data, count), transfer, lane));
        }
    }

    public override void DisconnectFromHost(NetError why, bool immediately = false)
    {
        if (!_connected)
        {
            return;
        }
        _connected = false;
        _host.ClientLeft(_seatId, why);
        _dropped = SimLoopbackHost.ErrorInfo(why, ours: true);
    }

    public override string? GetRawLobbyIdentifier() => null;

    internal void Deliver(SimLoopbackPacket packet)
    {
        if (_connected)
        {
            _inbox.Enqueue(packet);
        }
    }

    internal void HostDropped(NetErrorInfo info)
    {
        _connected = false;
        _inbox.Clear();
        _dropped = info;
    }
}

/// <summary>
/// The connection initializer JoinFlow.Begin takes, implemented at run time
/// against the game's own IClientConnectionInitializer (its Connect takes a
/// NetClientGameService in v0.107.1 and an INetClientGameService in
/// v0.111.0; the object passed is a NetClientGameService in both). v0.113.0
/// adds <c>Platform</c> and <c>GetHostId()</c>, which its JoinFlow reads
/// only to describe a failed join (JoinFailureExtraInfo).
/// </summary>
public class SimLoopbackConnector : DispatchProxy
{
    private SimLoopbackHost? _host;
    private ulong _netId;

    internal static IClientConnectionInitializer Create(SimLoopbackHost host, ulong netId)
    {
        IClientConnectionInitializer proxy = Create<IClientConnectionInitializer, SimLoopbackConnector>();
        var self = (SimLoopbackConnector)(object)proxy;
        self._host = host;
        self._netId = netId;
        return proxy;
    }

    protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
    {
        switch (targetMethod?.Name)
        {
            case "get_Platform":
                return PlatformType.None;
            case "GetHostId":
                return (ulong?)_host?.NetId;
        }
        if (targetMethod?.Name != "Connect" || args is not { Length: > 0 } || args[0] is not NetClientGameService service)
        {
            throw new NotSupportedException($"SimLoopbackConnector.{targetMethod?.Name}");
        }
        var client = new SimLoopbackClient(service, _host!, _netId);
        service.Initialize(client, PlatformType.None);
        return Task.FromResult(client.Connect());
    }

    public override string ToString() => $"SimLoopbackConnector({_netId})";
}

/// <summary>Builds a host NetHostGameService on the loopback transport.</summary>
internal static class SimLoopbackHosting
{
    private static readonly FieldInfo? NetHostField =
        typeof(NetHostGameService).GetField("_netHost", BindingFlags.NonPublic | BindingFlags.Instance);

    /// <summary>v0.111.0: <c>new NetHostGameService(PeerVersionInfo
    /// .LocalDefault())</c>; v0.107.1: <c>new NetHostGameService()</c>.</summary>
    public static (NetHostGameService Service, SimLoopbackHost Host) Start()
    {
        if (NetHostField == null)
        {
            throw new MissingFieldException(nameof(NetHostGameService), "_netHost");
        }
        object? version = LocalVersionInfo();
        var service = (NetHostGameService)(version != null
            ? Activator.CreateInstance(typeof(NetHostGameService), version)!
            : Activator.CreateInstance(typeof(NetHostGameService))!);
        var host = SimNetHosts.Create<SimLoopbackHost>(service, LocalHostId());
        NetHostField.SetValue(service, host);
        host.Start();
        return (service, host);
    }

    /// <summary>
    /// Puts AI partners next to the game's own host of a lobby the player
    /// opened (Steam, or ENet for fastmp): the service's NetHost becomes a
    /// <see cref="SimMixedHost"/> over it. Idempotent. Null when the service
    /// hosts on nothing, or on the loopback alone (a session of its own).
    /// </summary>
    public static SimLoopbackHost? AttachToGameHost(NetHostGameService service)
    {
        if (NetHostField == null)
        {
            throw new MissingFieldException(nameof(NetHostGameService), "_netHost");
        }
        switch (NetHostField.GetValue(service))
        {
            case SimMixedHost mixed:
                return mixed.Partners;
            case SimLoopbackHost:
                return null;
            case NetHost game when game.IsConnected:
                var partners = SimNetHosts.Create<SimLoopbackHost>(service, game.NetId);
                NetHostField.SetValue(service, SimNetHosts.Create<SimMixedHost>(service, game, partners));
                partners.Start();
                return partners;
            default:
                return null;
        }
    }

    /// <summary>
    /// The host's net id in a session of its own: the player's Steam id. The
    /// run is saved under its players' ids, and "Load" checks the save
    /// against the local Steam id (RunManager.CanonicalizeSave), so a run
    /// saved under the null platform's id 1 could never be continued (the
    /// game sets such a save aside as broken). Without Steam: 1.
    /// </summary>
    private static ulong LocalHostId()
    {
        try
        {
            if (!SteamInitializer.Initialized)
            {
                return SimLoopbackHost.HostId;
            }
            ulong steam = PlatformUtil.GetLocalPlayerId(PlatformType.Steam);
            return steam != 0 ? steam : SimLoopbackHost.HostId;
        }
        catch (Exception)
        {
            return SimLoopbackHost.HostId;
        }
    }

    /// <summary>The local PeerVersionInfo (a v0.111.0-only type), or null.</summary>
    public static object? LocalVersionInfo()
    {
        Type? versionType = typeof(NetClientGameService).Assembly.GetType(typeof(NetClientGameService).Namespace + ".PeerVersionInfo");
        return versionType?.GetMethod("LocalDefault", BindingFlags.Public | BindingFlags.Static)?.Invoke(null, null);
    }
}

/// <summary>
/// Concrete <see cref="SimLoopbackHost"/> / <see cref="SimMixedHost"/> types,
/// emitted at run time. NetHost.DisconnectClient is
/// <c>(ulong, NetError, bool now = false)</c> on v0.107.1 / v0.111.0 and
/// <c>(ulong, NetError, bool now = false, string? debugReason = null)</c> on
/// v0.113.0 (Multiplayer.Transport/NetHost.cs:74). The emitted subclass
/// overrides whichever the running game declares and forwards it to the
/// host's <c>DisconnectClientCore(ulong, NetError, bool)</c>; the extra debug
/// reason is dropped. Any OTHER abstract member left open means the game
/// changed NetHost again: Create then throws, which turns off AI partners
/// only (the caller logs it), instead of a TypeLoadException wherever the
/// host type is touched.
/// </summary>
public static class SimNetHosts
{
    // Guarded by Gate: DefineType throws on a second type of the same name in
    // one module, so every build of a host type goes through the cache.
    private static readonly object Gate = new();
    private static readonly Dictionary<Type, Type> Concrete = new();
    private static ModuleBuilder? _module;

    private static readonly Lazy<MethodInfo?> NetHostDisconnect =
        new(() => DisconnectMethod(typeof(NetHost)));

    /// <summary><c>new T(args)</c> for one of the abstract hosts.</summary>
    internal static T Create<T>(params object[] args) where T : NetHost =>
        (T)Activator.CreateInstance(Build(typeof(T)), args)!;

    /// <summary><c>host.DisconnectClient(peer, why, now)</c> on any build.</summary>
    internal static void DisconnectClient(NetHost host, ulong peer, NetError why, bool now)
    {
        MethodInfo m = NetHostDisconnect.Value
            ?? throw new MissingMethodException(nameof(NetHost), nameof(NetHost.DisconnectClient));
        ParameterInfo[] p = m.GetParameters();
        var args = new object?[p.Length];
        args[0] = peer;
        args[1] = why;
        args[2] = now;
        for (int i = 3; i < p.Length; i++) args[i] = p[i].DefaultValue;
        try
        {
            m.Invoke(host, args);
        }
        catch (TargetInvocationException ex) when (ex.InnerException != null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
        }
    }

    /// <summary>NetHost's abstract DisconnectClient: (ulong, NetError, bool, optional...).</summary>
    private static MethodInfo? DisconnectMethod(Type type) =>
        type.GetMethods(BindingFlags.Instance | BindingFlags.Public)
            .FirstOrDefault(m =>
            {
                ParameterInfo[] p = m.GetParameters();
                return m.Name == nameof(NetHost.DisconnectClient)
                    && m.IsAbstract
                    && p.Length >= 3
                    && p[0].ParameterType == typeof(ulong)
                    && p[1].ParameterType == typeof(NetError)
                    && p[2].ParameterType == typeof(bool)
                    && p.Skip(3).All(x => x.IsOptional);
            });

    /// <summary>The concrete subclass of <paramref name="abstractHost"/>,
    /// emitted once and cached (the wire smoke test calls this too).</summary>
    private static Type Build(Type abstractHost)
    {
        lock (Gate)
        {
            if (!Concrete.TryGetValue(abstractHost, out Type? concrete))
            {
                concrete = Emit(abstractHost);
                Concrete[abstractHost] = concrete;
            }
            return concrete;
        }
    }

    private static Type Emit(Type abstractHost)
    {
        _module ??= AssemblyBuilder
            .DefineDynamicAssembly(new AssemblyName("SpireMentor.NetHosts"), AssemblyBuilderAccess.Run)
            .DefineDynamicModule("SpireMentor.NetHosts");
        TypeBuilder tb = _module.DefineType(
            abstractHost.Name + "Impl",
            TypeAttributes.Public | TypeAttributes.Sealed | TypeAttributes.Class,
            abstractHost);

        foreach (ConstructorInfo ctor in abstractHost.GetConstructors(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
        {
            if (ctor.IsPrivate)
                continue;
            Type[] types = ctor.GetParameters().Select(p => p.ParameterType).ToArray();
            ConstructorBuilder cb = tb.DefineConstructor(MethodAttributes.Public, CallingConventions.Standard, types);
            ILGenerator il = cb.GetILGenerator();
            il.Emit(OpCodes.Ldarg_0);
            for (int i = 1; i <= types.Length; i++) il.Emit(OpCodes.Ldarg, (short)i);
            il.Emit(OpCodes.Call, ctor);
            il.Emit(OpCodes.Ret);
        }

        MethodInfo core = abstractHost.GetMethod("DisconnectClientCore", new[] { typeof(ulong), typeof(NetError), typeof(bool) })
            ?? throw new MissingMethodException(abstractHost.Name, "DisconnectClientCore");
        foreach (MethodInfo open in abstractHost.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                     .Where(m => m.IsAbstract))
        {
            if (open != DisconnectMethod(abstractHost))
            {
                throw new NotSupportedException(
                    $"Spire Mentor does not know this game build: NetHost.{open.Name} is abstract and not implemented by {abstractHost.Name}");
            }
            Type[] types = open.GetParameters().Select(p => p.ParameterType).ToArray();
            MethodBuilder mb = tb.DefineMethod(
                open.Name,
                MethodAttributes.Public | MethodAttributes.Virtual | MethodAttributes.HideBySig,
                open.ReturnType,
                types);
            ILGenerator il = mb.GetILGenerator();
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, core);
            il.Emit(OpCodes.Ret);
            tb.DefineMethodOverride(mb, open);
        }
        return tb.CreateType()!;
    }
}
