"""FIOR Python library - basic usage.

This example walks the whole loop: read a model card, list sites, fetch each
site's delta-eta blob, compose a prior, and score peers against a local
likelihood.  It assumes a relay and Blossom server are configured; the exact
same flow is what the CLI's `fetch`, `compose`, and `score` commands run.
"""

import numpy as np

from fior import Client, Eta, Site

RELAY = "wss://relay.example.com"
BLOSSOM = ["https://blossom.example.com"]
MODEL = "pump-failure-v1"
# dims would normally come from parsing the ONNX graph client-side.
INITIALIZER_DIMS = {"fc1.weight": 256, "fc1.bias": 256}


def main():
    c = Client(relay_url=RELAY, blossom_servers=BLOSSOM, private_key="<your-hex-key>")

    card = c.read_model_card(MODEL, initializer_dims=INITIALIZER_DIMS)
    if card is None:
        print("no model card found")
        return
    print(f"Model: {card.title} (v{card.version}), groups: "
          f"{[(g.name, g.family, g.dim) for g in card.groups]}")

    descs = c.list_sites(MODEL)
    members = []
    for d in descs:
        delta = c.fetch_site_params(d, MODEL)
        if delta:
            members.append(Site(author=d.author, model_id=MODEL,
                                model_version=d.model_version, delta_eta=delta,
                                event_id=d.event_id))
    print(f"{len(members)} sites resolved")

    # trust table: currently held p values (defaults to 0.5 for strangers)
    p_table = {d.author: {g.name: c.trust.get_p(d.author, g.name) for g in card.groups}
               for d in descs}
    prior = c.compose_prior(card, members, p_table)
    print("composed prior groups:", {g: e.implied_mean() for g, e in prior.items()})

    # local likelihood: whatever the local fit produced (posterior - prior)
    lik = Eta("mvnormal", np.array([0.1, -0.2]), np.array([[-0.5, 0.0], [0.0, -0.5]]))
    posterior = {g: prior[g] + lik for g in prior}
    new_p = c.score_peers(card, prior, posterior, members, p_table)
    print("scored peers:", {a: {g: round(v, 4) for g, v in gr.items()}
                            for a, gr in new_p.items()})


if __name__ == "__main__":
    main()