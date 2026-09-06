// REAL-WORLD ACTION -- validated operations on REAL assets.
//
// A policy decision is advisory input. A RealActionPlan names physical
// targets (or none, for machine-local ops) and carries the approval
// verdict. Blocked plans (phantom target / no physical footprint) must
// be rendered as refusals and can never execute.

namespace CyberMarl.Deployment.Core;

/// <summary>Validated operation on the real network (or a loud refusal).</summary>
public sealed record RealActionPlan(
    string AssetId,
    string Hostname,
    IReadOnlyList<string> Ips,
    string Operation, // validator TRANSLATION value, e.g. reimage_host
    IReadOnlyDictionary<string, string> Params,
    string Risk,      // safe | elevated | destructive
    bool RequiresApproval,
    bool Blocked,
    string BlockReason,
    PolicyProvenance Provenance)
{
    public string Describe()
    {
        if (Blocked) return $"BLOCKED: {BlockReason}";
        var target = !string.IsNullOrEmpty(Hostname) ? Hostname
            : !string.IsNullOrEmpty(AssetId) ? AssetId : "local";
        var approval = RequiresApproval ? ", needs-approval" : "";
        return $"{Operation} on {target} [{Risk}{approval}]";
    }
}

/// <summary>Which policy decision produced this plan (audit trail).</summary>
public sealed record PolicyProvenance(
    int AgentId,
    int ActionIndex,
    string Command,
    string Kind,
    string PolicyTarget,
    string PolicyLabel,
    string MappingDigest);
