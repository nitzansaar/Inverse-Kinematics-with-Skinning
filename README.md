# Inverse Kinematics with Skinning


## Linear Blend Skinning

Skinning is the process of deforming a surface mesh to follow an underlying skeleton. Each vertex on the mesh has a set of weights that describe how much each nearby joint influences its position. When a joint moves, every vertex it influences gets pulled along proportionally.

The formula is straightforward: the new position of a vertex is a weighted sum of where each influencing joint would place it independently:

```
p' = sum_i ( w_i * skinTransform[i] * p_rest )
```

The weights `w_i` sum to 1 for each vertex, so the result is a convex blend. Vertices near the center of a bone (high single-joint weight) move rigidly with it, while vertices near a joint (spread across multiple joints) stretch and blend between the influences — which is what produces the smooth deformation you see at elbows, shoulders, and other articulation points.

The skin transform for each joint is `globalTransform * invRestGlobalTransform`: the inverse rest transform "un-does" the bind pose, and the current global transform re-applies the joint's new orientation. In practice this means the deformation is always measured relative to where the skeleton started.

A known limitation of linear blend skinning is "candy-wrapper" or "collapsing elbow" artifacts — when a joint rotates more than ~90 degrees, the blended rotations can cause the mesh to pinch or collapse. More sophisticated methods (dual quaternion skinning, etc.) address this, but LBS remains the industry baseline due to its simplicity and GPU-friendliness.

## Forward Kinematics

Forward kinematics is how the skeleton's bone transforms are computed from joint angles. Each joint stores a local transform — its rotation and translation relative to its parent — and the global transform of any joint is found by chaining all the local transforms from the root down to that joint:

```
globalTransform[i] = globalTransform[parent] * localTransform[i]
```

The local rotation for each joint is composed as `R_jointOrientation * R_euler`, following the Maya convention used throughout this assignment. The joint orientation is a fixed offset baked in at rig export time (always applied in XYZ order), and the Euler rotation is what animators or IK solvers actually change at runtime.

One subtlety I encountered is the rotation order. Different joints can have different Euler rotation orders (XYZ, ZYX, etc.), and the order matters — applying the same three angles in a different sequence produces a different rotation. The skeleton config file stores the rotation order per joint, and the FK implementation has to respect it when building each local transform.

The FK computation has to process joints in a specific order: parents before children, so that a joint's parent global transform is ready before it's needed. This is handled by pre-computing a `jointUpdateOrder` at load time using a depth-first traversal from the root.

## Inverse Kinematics

Where FK goes from joint angles to bone positions, IK goes in the opposite direction: given desired positions for certain end-effector joints (the "handles"), find joint angles that put those handles where you want them. This is a fundamentally harder problem — the mapping from angles to positions is nonlinear, and there are often infinitely many valid solutions.

The approach used here is Jacobian-based IK with Tikhonov regularization. The Jacobian J is the matrix of partial derivatives of the handle positions with respect to all joint angles — it describes, locally, how a small change in each angle moves each handle. Computing J by hand would be tedious and error-prone, so instead ADOL-C (automatic differentiation) is used: the FK function is written once in a templated form, recorded as a computation tape, and then ADOL-C evaluates the exact Jacobian numerically at any configuration.

Given J, the IK step is solved as:

```
(J^T J + alpha * I) * dtheta = J^T * dp
```

where `dp` is the gap between the current and target handle positions. The `alpha * I` term is Tikhonov regularization — it keeps the system well-conditioned when the Jacobian is rank-deficient (e.g., near singularities or when there are more degrees of freedom than constraints), preventing the solver from producing wild angle swings. The tradeoff is that the handles converge to their targets over multiple frames rather than in a single step, but this is visually fine and actually looks more natural.

This one-step-per-frame loop — evaluate FK, compute Jacobian, solve the linear system, update angles — is what makes the interactive dragging work in real time.
