// Genuinely-real local discovery in C#: hostname, interface addresses,
// MACs and OS of THIS machine (inbox APIs only). A single PC therefore
// presents exactly one real asset. Process/connection inventory and
// intrusion verdicts are NOT fabricated here -- they arrive as real
// telemetry batches (EDR/SIEM adapters) or as explicit unknown.

using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Runtime.InteropServices;

namespace CyberMarl.Deployment.Core;

/// <summary>Discovers the deployment host itself (real sensing).</summary>
public static class LocalDiscovery
{
    public static string AssetIdFor(string hostname) =>
        "host:" + (hostname ?? "unknown").Trim().ToLowerInvariant();

    public static (RealTopologyDto Topology, RealBatchDto Batch) DiscoverThisMachine()
    {
        var hostname = SafeHostname();
        var assetId = AssetIdFor(hostname);
        var ips = LocalIpv4Addresses();
        var macs = LocalMacs();
        var processes = LocalProcessNames();
        var listeners = LocalTcpListeners();
        var now = DateTimeOffset.UtcNow;

        var topology = new RealTopologyDto(
            new List<RealAssetDto>
            {
                new(assetId, hostname, ips, macs,
                    RuntimeInformation.OSDescription.Trim(),
                    Environment.OSVersion.VersionString,
                    "", "host-only", "covered-local",
                    "sensed on this machine"),
            },
            new List<RealSegmentDto>
            {
                new("host-only", "This machine (host-only)", "host-only", new List<string>()),
            },
            "csharp-local-discovery",
            now.ToUnixTimeSeconds());

        // Genuinely observed inventory (processes + TCP states). Service
        // enumeration and intrusion verdicts need a telemetry adapter:
        // unknown stays unknown, never synthesized.
        var batch = new RealBatchDto(
            now.ToUnixTimeSeconds(),
            "csharp-local-discovery",
            new Dictionary<string, HostTelemetryDto>
            {
                [assetId] = new(processes, listeners, new List<string>(), true,
                    new List<SecurityEventDto>()),
            },
            $"local discovery for {hostname}",
            new List<Dictionary<string, string>>
            {
                new() { ["key"] = assetId, ["error"] = "service inventory + intrusion verdicts need a telemetry adapter" },
            });
        return (topology, batch);
    }

    /// <summary>Real process names (inbox API, cross-platform).</summary>
    public static List<string> LocalProcessNames()
    {
        var names = new SortedSet<string>(StringComparer.Ordinal);
        try
        {
            foreach (var process in System.Diagnostics.Process.GetProcesses())
            {
                try
                {
                    var name = process.ProcessName?.Trim();
                    if (!string.IsNullOrEmpty(name)) names.Add(name);
                }
                catch { /* raced exit; skip */ }
                finally { try { process.Dispose(); } catch { } }
            }
        }
        catch { /* degraded: caller reports unknown */ }
        return names.ToList();
    }

    /// <summary>Real TCP states as "tcp:port[-&gt;peer]" (inbox API).</summary>
    public static List<string> LocalTcpListeners()
    {
        var found = new SortedSet<string>(StringComparer.Ordinal);
        try
        {
            var props = System.Net.NetworkInformation.IPGlobalProperties
                .GetIPGlobalProperties();
            foreach (var listener in props.GetActiveTcpListeners())
                found.Add($"tcp:{listener.Port}");
            foreach (var connection in props.GetActiveTcpConnections())
            {
                if (connection.State == System.Net.NetworkInformation.TcpState.Established &&
                    connection.RemoteEndPoint is not null)
                    found.Add($"tcp:{connection.LocalEndPoint.Port}->{connection.RemoteEndPoint.Address}");
            }
        }
        catch { /* degraded */ }
        return found.ToList();
    }

    private static string SafeHostname()
    {
        try { return Dns.GetHostName().Trim(); }
        catch { return "unknown"; }
    }

    private static List<string> LocalIpv4Addresses()
    {
        var result = new List<string>();
        try
        {
            foreach (var nic in NetworkInterface.GetAllNetworkInterfaces())
            {
                if (nic.OperationalStatus != OperationalStatus.Up) continue;
                if (nic.NetworkInterfaceType == NetworkInterfaceType.Loopback) continue;
                var props = nic.GetIPProperties();
                foreach (var unicast in props.UnicastAddresses)
                {
                    if (unicast.Address.AddressFamily != AddressFamily.InterNetwork) continue;
                    var text = unicast.Address.ToString();
                    if (IPAddress.IsLoopback(unicast.Address)) continue;
                    if (!result.Contains(text)) result.Add(text);
                }
            }
        }
        catch { /* degraded: caller reports unknown, never fiction */ }
        result.Sort(StringComparer.Ordinal);
        return result;
    }

    private static List<string> LocalMacs()
    {
        var result = new List<string>();
        try
        {
            foreach (var nic in NetworkInterface.GetAllNetworkInterfaces())
            {
                if (nic.OperationalStatus != OperationalStatus.Up) continue;
                if (nic.NetworkInterfaceType == NetworkInterfaceType.Loopback) continue;
                var bytes = nic.GetPhysicalAddress()?.GetAddressBytes();
                if (bytes is null || bytes.Length != 6) continue;
                if (bytes.All(b => b == 0)) continue;
                var text = string.Join(":", bytes.Select(b => b.ToString("x2")));
                if (!result.Contains(text)) result.Add(text);
            }
        }
        catch { /* degraded */ }
        result.Sort(StringComparer.Ordinal);
        return result;
    }
}
