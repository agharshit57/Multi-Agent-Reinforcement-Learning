// REAL ASSET / REAL TOPOLOGY -- what physically exists.
//
// These types never carry CC4 zone names, slot ids, observation dims or
// action ids. Identity is AssetId ("host:<lower-hostname>"); telemetry
// keys are AssetIds. The CC4/policy world lives in PolicyMapping.cs and
// meets these types only inside the mapping DTOs from the sidecar.

namespace CyberMarl.Deployment.Core;

/// <summary>One physical machine in the deployment.</summary>
public sealed record RealAsset(
    string AssetId,
    string Hostname,
    IReadOnlyList<string> Ips,
    IReadOnlyList<string> Macs,
    string OsName,
    string OsVersion,
    string Role,
    string SegmentId,
    string Coverage = "unknown",
    string CoverageDetail = "",
    IReadOnlyList<string>? Services = null)
{
    /// <summary>Best address for display: first non-loopback, else first.</summary>
    public string PrimaryIp
    {
        get
        {
            foreach (var ip in Ips)
            {
                if (!IsLoopback(ip)) return ip;
            }
            return Ips.Count > 0 ? Ips[0] : "no address";
        }
    }

    public static bool IsLoopback(string address)
    {
        if (string.IsNullOrWhiteSpace(address)) return true;
        var text = address.Split('%')[0].Trim().ToLowerInvariant();
        if (text == "localhost" || text == "::1") return true;
        return text.StartsWith("127.");
    }

    public string Describe() =>
        $"{Hostname} [{AssetId}] @ {PrimaryIp} ({(string.IsNullOrEmpty(OsName) ? "unknown OS" : OsName)})";
}

/// <summary>One real network segment (site switching/L3 truth, not a CC4 zone).</summary>
public sealed record RealSegment(
    string SegmentId,
    string Name,
    string Kind, // host-only | lan | unknown
    IReadOnlyList<string> Cidrs);

/// <summary>One labeled relationship between real assets.</summary>
public sealed record RealLink(string AssetId, string Peer, string Via);

/// <summary>Physical inventory at a point in time.</summary>
public sealed record RealTopology(
    IReadOnlyList<RealAsset> Assets,
    IReadOnlyList<RealSegment> Segments,
    string Source,
    DateTimeOffset DiscoveredAt,
    IReadOnlyList<RealLink>? Links = null)
{
    public RealAsset? Find(string assetId) =>
        Assets.FirstOrDefault(a => a.AssetId == assetId);
}
