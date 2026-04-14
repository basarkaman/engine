/// CPU-side frustum built from a view-projection matrix.
/// Used to cull meshes and instances before submitting draw calls.
pub struct Frustum {
    /// 6 half-space planes. A point P is *inside* plane i when
    ///   planes[i].x*P.x + planes[i].y*P.y + planes[i].z*P.z + planes[i].w >= 0
    /// Normals are unit-length.
    planes: [cgmath::Vector4<f32>; 6],
}

impl Frustum {
    /// Build from a combined view-projection matrix that maps to
    /// wgpu's NDC (x,y in [-1,1], z in [0,1]).
    pub fn from_vp(vp: &cgmath::Matrix4<f32>) -> Self {
        // cgmath stores matrices column-major: vp[col][row].
        // Extract the four rows as homogeneous vectors.
        let row = |r: usize| {
            cgmath::Vector4::new(vp[0][r], vp[1][r], vp[2][r], vp[3][r])
        };
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));

        // Gribb/Hartmann plane extraction, adapted for z in [0,1]:
        //   left:   r3 + r0    right:  r3 - r0
        //   bottom: r3 + r1    top:    r3 - r1
        //   near:   r2         (z >= 0)
        //   far:    r3 - r2    (z <= 1)
        let mut planes = [
            r3 + r0,
            r3 - r0,
            r3 + r1,
            r3 - r1,
            r2,
            r3 - r2,
        ];

        for p in &mut planes {
            let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            if len > 1e-6 {
                *p /= len;
            }
        }

        Self { planes }
    }

    /// Returns `true` if the sphere is **entirely outside** the frustum (safe to cull).
    #[inline]
    pub fn cull_sphere(&self, center: cgmath::Point3<f32>, radius: f32) -> bool {
        for p in &self.planes {
            let signed_dist = p.x * center.x + p.y * center.y + p.z * center.z + p.w;
            if signed_dist < -radius {
                return true;
            }
        }
        false
    }
}

impl Default for Frustum {
    /// Identity frustum — everything is considered visible.
    fn default() -> Self {
        Self {
            planes: [
                cgmath::Vector4::new( 1.0,  0.0,  0.0,  1.0),
                cgmath::Vector4::new(-1.0,  0.0,  0.0,  1.0),
                cgmath::Vector4::new( 0.0,  1.0,  0.0,  1.0),
                cgmath::Vector4::new( 0.0, -1.0,  0.0,  1.0),
                cgmath::Vector4::new( 0.0,  0.0,  1.0,  1.0),
                cgmath::Vector4::new( 0.0,  0.0, -1.0,  1.0),
            ],
        }
    }
}
