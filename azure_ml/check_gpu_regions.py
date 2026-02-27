"""Find cheapest GPU VM across allowed regions for Azure for Students."""
from azure.identity import VisualStudioCodeCredential
from azure.mgmt.compute import ComputeManagementClient

cred = VisualStudioCodeCredential()
client = ComputeManagementClient(cred, "02b4b69d-dd14-4e79-b35f-de906edb6b15")

# Regions that Azure for Students typically allows
regions = [
    "westeurope", "northeurope", "swedencentral", "norwayeast",
    "eastus", "eastus2", "centralus", "westus2", "westus3",
    "uksouth", "francecentral", "germanywestcentral",
]

print("Checking GPU VM availability across regions...\n")
found = {}
for region in regions:
    gpu_vms = []
    try:
        for s in client.resource_skus.list(filter=f"location eq '{region}'"):
            if s.resource_type == "virtualMachines" and s.name and (
                "NC" in s.name or "ND" in s.name
            ):
                gpu_vms.append(s.name)
    except Exception:
        pass
    if gpu_vms:
        found[region] = sorted(set(gpu_vms))
        print(f"{region}: {found[region]}")
    else:
        print(f"{region}: (none)")

print("\n--- Summary ---")
t4_regions = {r: vms for r, vms in found.items() if "Standard_NC4as_T4_v3" in vms}
if t4_regions:
    print(f"T4 (NC4as_T4_v3) available in: {list(t4_regions.keys())}")
else:
    print("T4 (NC4as_T4_v3) not found. Available GPU VMs per region:")
    for r, vms in found.items():
        print(f"  {r}: {vms}")

