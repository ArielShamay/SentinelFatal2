"""Find which region actually ALLOWS storage account creation."""
from azure.identity import VisualStudioCodeCredential
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.resource import ResourceManagementClient
import time

SUBSCRIPTION = "02b4b69d-dd14-4e79-b35f-de906edb6b15"
TEST_RG      = "sentinelfatal2-test-rg"

cred        = VisualStudioCodeCredential()
rg_client   = ResourceManagementClient(cred, SUBSCRIPTION)

# Regions to try (ordered by latency from Israel)
regions = [
    "westeurope", "northeurope", "uksouth", "ukwest",
    "francecentral", "germanywestcentral", "swedencentral",
    "eastus", "eastus2", "centralus", "westus2",
    "israelcentral",
]

print("Testing which regions allow Storage Account creation...\n")
found_region = None

for region in regions:
    rg_name = f"{TEST_RG}-{region}"
    try:
        # Create test RG
        rg_client.resource_groups.create_or_update(rg_name, {"location": region})
    except Exception as e:
        print(f"  {region}: RG creation failed — {e}")
        continue

    # Try to create a Storage Account
    storage_client = StorageManagementClient(cred, SUBSCRIPTION)
    acct_name = f"sttest{int(time.time()) % 100000}"
    try:
        poller = storage_client.storage_accounts.begin_create(
            rg_name, acct_name,
            {
                "sku": {"name": "Standard_LRS"},
                "kind": "StorageV2",
                "location": region,
            }
        )
        poller.result(timeout=60)
        print(f"  {region}: ✓ ALLOWED")
        found_region = region
        # Clean up
        try:
            storage_client.storage_accounts.delete(rg_name, acct_name)
        except Exception:
            pass
        rg_client.resource_groups.begin_delete(rg_name)
        break
    except Exception as e:
        err = str(e)
        if "RequestDisallowedByAzure" in err or "disallowed" in err.lower():
            print(f"  {region}: ✗ POLICY BLOCKED")
        else:
            print(f"  {region}: ? Error — {err[:100]}")
    finally:
        try:
            rg_client.resource_groups.begin_delete(rg_name)
        except Exception:
            pass

if found_region:
    print(f"\n>>> Use region: {found_region} <<<")
else:
    print("\n>>> No allowed region found. Contact Azure support to lift regional policy. <<<")
