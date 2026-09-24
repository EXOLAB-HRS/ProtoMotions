"""Restore physical-neighbor exclusions across collider-free joint frames.

Owner: human_model_v3.1. Restores neighbors and preserves legacy filters.
Hand/Chest and opposite-foot contacts remain enabled; some legacy cross-limb
Knee/Ankle/Toe exclusions are deliberately retained.
The source v2 asset is immutable; the generated USD layer authors filters only.
"""
from pathlib import Path
import hashlib
import json


def inspect_stage(stage, root=None):
    from pxr import Usd, UsdPhysics
    colliding = set()
    parents = {}
    excluded = set()
    adjacent = set()
    prims = stage.Traverse() if root is None else Usd.PrimRange(stage.GetPrimAtPath(root))
    for prim in prims:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            owner = prim
            while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if not owner:
                raise ValueError(f'Collider without rigid body: {prim.GetPath()}')
            colliding.add(str(owner.GetPath()))
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            excluded.update(tuple(sorted((str(prim.GetPath()), str(target))))
                            for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets())
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not a or not b:
                continue
            parent, child = str(a[0]), str(b[0])
            if child in parents:
                raise ValueError(f'Expected articulation tree, multiple parents: {child}')
            parents[child] = parent
            if not joint.GetCollisionEnabledAttr().Get():
                adjacent.add(tuple(sorted((parent, child))))
    physical_neighbors = set()
    for child in sorted(colliding):
        parent = parents.get(child)
        visited = {child}
        while parent and parent not in colliding:
            if parent in visited:
                raise ValueError('Joint hierarchy cycle')
            visited.add(parent)
            parent = parents.get(parent)
        if parent:
            physical_neighbors.add(tuple(sorted((parent, child))))
    return dict(colliding=colliding, excluded=excluded, adjacent=adjacent,
                physical_neighbors=physical_neighbors)


def validate_collision_asset(source, corrected):
    from pxr import Usd
    a, b = Usd.Stage.Open(str(source)), Usd.Stage.Open(str(corrected))
    if a is None or b is None:
        raise ValueError('Collision asset could not be opened')
    before, after = inspect_stage(a), inspect_stage(b)
    expected = before['excluded'] | before['physical_neighbors']
    if after['excluded'] != expected:
        raise ValueError('Collision revision has missing or over-broad exclusions')
    if before['colliding'] != after['colliding'] or before['adjacent'] != after['adjacent']:
        raise ValueError('Collision revision changed collider or joint topology')
    if {str(p.GetPath()) for p in a.Traverse()} != {str(p.GetPath()) for p in b.Traverse()}:
        raise ValueError('Collision revision changed prim set')
    for prim in a.Traverse():
        other = b.GetPrimAtPath(prim.GetPath())
        if prim.GetTypeName() != other.GetTypeName():
            raise ValueError('Collision revision changed prim type')
        for attr in prim.GetAttributes():
            if attr.Get() != other.GetAttribute(attr.GetName()).Get():
                raise ValueError(f'Collision revision changed attribute: {attr.GetPath()}')
        for rel in prim.GetRelationships():
            if rel.GetName() != 'physics:filteredPairs' and rel.GetTargets() != other.GetRelationship(rel.GetName()).GetTargets():
                raise ValueError(f'Collision revision changed relationship: {rel.GetPath()}')
    short = lambda pairs: [list(map(lambda p: p.rsplit('/', 1)[-1], pair)) for pair in sorted(pairs)]
    return dict(model_id='human_model_v3.1', collider_count=len(before['colliding']),
                source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                overlay_sha256=hashlib.sha256(Path(corrected).read_bytes()).hexdigest(),
                physical_neighbor_pairs=short(before['physical_neighbors']),
                restored_pairs=short(before['physical_neighbors'] - before['excluded'] - before['adjacent']),
                effective_exclusion_count=len(after['excluded']),
                unchanged='all source attributes and non-filter relationships')


def build():
    from pxr import Sdf, Usd, UsdPhysics
    import os
    root = Path(__file__).resolve().parent
    source = root.parent / 'human_model_v2/assets/human_model_v2.usda'
    destination = root / 'assets/human_model_v3_1.usda'
    source_stage = Usd.Stage.Open(str(source))
    info = inspect_stage(source_stage)
    stage = Usd.Stage.CreateInMemory()
    # Generate an override layer, rather than a second full copy of the plant.
    pairs = info['physical_neighbors'] | info['excluded']
    for owner in sorted({p[0] for p in pairs}):
        # Preserve the source's outgoing relationships (pairs are undirected).
        targets = set(str(t) for t in UsdPhysics.FilteredPairsAPI(source_stage.GetPrimAtPath(owner)).GetFilteredPairsRel().GetTargets())
        targets.update(p[1] for p in pairs if p[0] == owner)
        prim = stage.OverridePrim(owner)
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(sorted(targets))
    stage.SetMetadata('upAxis', source_stage.GetMetadata('upAxis'))
    output_layer = Sdf.Layer.CreateAnonymous()
    output_layer.TransferContent(stage.GetRootLayer())
    output_layer.subLayerPaths = [os.path.relpath(source, destination.parent)]
    output_layer.defaultPrim = source_stage.GetDefaultPrim().GetName()
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_layer.Export(str(destination))
    return validate_collision_asset(source, destination)


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
