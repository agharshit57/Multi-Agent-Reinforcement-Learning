// POLICY REPRESENTATION (technical view only).
//
// A PolicyMapping is the auditable record of how real assets were placed
// into the frozen 112-slot CC4 geometry for one cycle. Slots without a
// physical asset are UNPOPULATED padding -- they must be rendered only
// inside a "policy mapping (technical)" drawer, never as infrastructure.

namespace CyberMarl.Deployment.Core;

/// <summary>One real-asset-to-policy-slot binding.</summary>
public sealed record PolicyBinding(
    string AssetId,
    string Hostname,
    IReadOnlyList<string> Ips,
    string Agent,      // blue_agent_N (policy-side owner, technical)
    string PolicySlot, // CC4 slot name (technical, never infrastructure)
    string SlotKey);

/// <summary>Auditable real-to-policy placement for one cycle.</summary>
public sealed record PolicyMapping(
    IReadOnlyList<PolicyBinding> Bindings,
    IReadOnlyList<string> UnpopulatedSlots,
    IReadOnlyList<string> OverflowAssetIds,
    string Digest)
{
    public string? AssetOfSlot(string policySlot) =>
        Bindings.FirstOrDefault(b => b.PolicySlot == policySlot)?.AssetId;

    public (string Agent, string Slot)? SlotOfAsset(string assetId)
    {
        var binding = Bindings.FirstOrDefault(b => b.AssetId == assetId);
        return binding is null ? null : (binding.Agent, binding.PolicySlot);
    }

    public bool IsUnpopulated(string policySlot) =>
        UnpopulatedSlots.Contains(policySlot);

    public string Summary() =>
        $"policy mapping {Digest}: {Bindings.Count} real asset(s) bound, " +
        $"{UnpopulatedSlots.Count} unpopulated policy slots, " +
        $"{OverflowAssetIds.Count} overflow (inventory only)";
}
