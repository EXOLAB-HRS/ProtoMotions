"""human_model_v3.5 asset check: the composed v3.5 stage equals the v3.1 collision stage except for body mass properties."""
MASS_ATTRS = {"physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes"}


def validate_inertia_asset(source, corrected):
    from pxr import Usd
    a, b = Usd.Stage.Open(str(source)), Usd.Stage.Open(str(corrected))
    if a is None or b is None:
        raise ValueError("Inertia asset could not be opened")
    if {str(p.GetPath()) for p in a.Traverse()} != {str(p.GetPath()) for p in b.Traverse()}:
        raise ValueError("Inertia revision changed the prim set")
    changed, total = [], 0.0
    for prim in a.Traverse():
        other = b.GetPrimAtPath(prim.GetPath())
        if prim.GetTypeName() != other.GetTypeName():
            raise ValueError("Inertia revision changed a prim type")
        for attr in prim.GetAttributes():
            if attr.Get() != other.GetAttribute(attr.GetName()).Get():
                if attr.GetName() not in MASS_ATTRS:
                    raise ValueError(f"Inertia revision changed a non-mass attribute: {attr.GetPath()}")
                changed.append(str(attr.GetPath()))
        for rel in prim.GetRelationships():
            if rel.GetTargets() != other.GetRelationship(rel.GetName()).GetTargets():
                raise ValueError(f"Inertia revision changed a relationship: {rel.GetPath()}")
        m = a.GetPrimAtPath(prim.GetPath()).GetAttribute("physics:mass")
        if m and m.Get() is not None:
            total += other.GetAttribute("physics:mass").Get() - m.Get()
    if abs(total) > 1e-3:
        raise ValueError(f"Inertia revision changed the total mass by {total} kg")
    return dict(model_id="human_model_v3.5", changed_attributes=len(changed))
