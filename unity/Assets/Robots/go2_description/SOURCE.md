# go2_description (official Unitree Go2 URDF)

Source: https://github.com/unitreerobotics/unitree_ros, `robots/go2_description`,
commit a3b70cae6fd4a82c0e1ece633d5c6f97e88c9d76 (BSD-3-Clause).
Copied unchanged: `urdf/go2_description.urdf` (moved to `Assets/Robots/go2_description.urdf`: the URDF Importer resolves
`package://go2_description/...` relative to the URDF's folder), `dae/` (meshes the URDF references), `package.xml`.
Left out: `meshes/` (byte-identical to `dae/`), xacro, launch, config, collision-model PNGs.
