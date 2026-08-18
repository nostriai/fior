"""Basic usage example for FIOR Python library."""

import asyncio
import numpy as np
from fior import Client, Eta, Site, ModelCard, Group


async def main():
    """Example: fetch model, sites, compose prior."""
    
    # Create client
    client = Client(
        relay_url="wss://relay.example.com",
        blossom_servers=["https://blossom.example.com"],
    )
    
    # Fetch a model card
    model = await client.fetch_model_card("pump-failure-v1")
    if model:
        print(f"Model: {model.title} (v{model.version})")
        print(f"Groups: {[g.name for g in model.groups]}")
    
    # Fetch sites for the model
    sites = await client.fetch_sites("pump-failure-v1")
    print(f"Found {len(sites)} sites")
    
    # Compose prior from trusted peers
    if sites and model:
        # Simple trust: full weight for first site
        p_values = [0.9] + [0.5] * (len(sites) - 1)
        prior = client.compose_prior(model.eta0, sites, p_values)
        print(f"Composed prior: dim={prior.dim}")
    
    # Compute p for a peer
    if sites:
        site = sites[0]
        
        # Build cavity (prior without this site)
        other_sites = sites[1:]
        cavity = client.compose_prior(model.eta0, other_sites, [0.5] * len(other_sites))
        
        # Local data contribution (placeholder)
        local_likelihood = Eta(h=np.array([0.1]), Lam=np.array([-0.1]))
        
        # Compute p
        p = client.compute_p(site, cavity, local_likelihood, beta=0.5)
        print(f"Computed p for {site.author[:8]}...: {p:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
