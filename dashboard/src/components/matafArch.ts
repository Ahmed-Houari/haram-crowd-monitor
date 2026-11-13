// The edge of the Mataf for the 3D twin, low detail on purpose (the crowd on the floor is the subject). Matched to the
// live broadcast frames and reference photos (docs/design/README.md §9, §11): a white marble floor, then the golden
// Ottoman arcade in a ring with a row of small white domes on top, then the grey multi-level Mataf expansion stepping
// up behind it. Clean silhouettes, no arches modelled one by one, no thin lines, no coplanar faces (every recess is
// a real offset, so nothing shimmers), no textures. No text anywhere. Plan in metres around the Kaaba.
import * as THREE from "three";

export const S = 0.116; // scene units per metre (same as ./kaaba.ts)
const m = (metres: number) => metres * S;

export const FLOOR_R = 62; // m: the floor map's radius (backend/floormap.py); the arcade's inner face stands at 60 m
const ARC_IN = 60, ARC_OUT = 66, ARC_H = 7.4, ARC_RECESS = 0.7; // Ottoman arcade ring
// the Mataf expansion, kept thin and low on purpose (2026-10-26, design freeze: the floor is the subject): two
// stepped levels behind the arcade (r0 = its face, m)
const EXP = [{ r0: 66, h: 6 }, { r0: 68.5, h: 10 }];
export const EXP_OUT = 76;
export const PLINTH_R = 79;
export const RING_MID = (66 + EXP_OUT) / 2; // m: where the camera markers stand, on the ring outside the floor
export const LEVEL = { first: EXP[0].h, band: EXP[1].h, roof: EXP[1].h };
const SEGS = 120;

/** Ring slab between two radii, from y0 to y1 metres. Materials: [top and bottom, walls]. */
function ring(r0: number, r1: number, y0: number, y1: number, mat: THREE.Material | THREE.Material[]) {
  const s = new THREE.Shape().absarc(0, 0, m(r1), 0, Math.PI * 2, false);
  s.holes = [new THREE.Path().absarc(0, 0, m(r0), 0, Math.PI * 2, true)];
  const geo = new THREE.ExtrudeGeometry(s, { depth: m(y1 - y0), bevelEnabled: false, curveSegments: SEGS / 4 });
  geo.rotateX(-Math.PI / 2);
  geo.translate(0, m(y0), 0);
  return new THREE.Mesh(geo, mat);
}

export interface MatafArch {
  group: THREE.Group;
  /** height (scene units) of the top at a plan point (scene units from the Kaaba): where camera masts stand */
  topAt: (x: number, z: number) => number;
  /** floor levels (scene units) a camera inside the galleries can stand on */
  floors: number[];
  /** points (scene units) the view keeps in frame */
  extent: THREE.Vector3[];
}

export function buildMatafArch(): MatafArch {
  const g = new THREE.Group();
  const lambert = (color: number) => new THREE.MeshLambertMaterial({ color });
  const marble = lambert(0xa9a7a1); // the white marble floor, dimmed for the night
  const gold = lambert(0xb48f58); // the Ottoman arcade's honey-gold stone, lit at night
  const goldShade = lambert(0x4b3b26); // its arches, in shadow
  const dome = lambert(0xd8d3c7); // its small white domes
  const grey = lambert(0x7d8285); // the Mataf expansion
  const greyTop = lambert(0x9a9fa1); // its terraces
  const roof = lambert(0x868b8d); // its roof ring
  const greyShade = lambert(0x3a4043); // its open gallery level, in shadow
  const base = lambert(0x0e1a1d);

  const plinth = new THREE.Mesh(new THREE.CylinderGeometry(m(PLINTH_R), m(PLINTH_R + 1), m(1), 96), base);
  plinth.position.y = -m(0.5) - 0.02;
  const floor = new THREE.Mesh(new THREE.CircleGeometry(m(ARC_OUT + 1), SEGS), marble);
  floor.rotation.x = -Math.PI / 2;
  g.add(plinth, floor);

  // the Ottoman arcade: the arches as a recessed dark band, the stone above them, the domes on top
  g.add(ring(ARC_IN + ARC_RECESS, ARC_OUT, 0.02, ARC_H - 2.4, [gold, goldShade]));
  g.add(ring(ARC_IN, ARC_OUT, ARC_H - 2.4, ARC_H, gold));
  const n = Math.round((2 * Math.PI * (ARC_IN + 3)) / 5.4); // one small dome per ~5.4 m bay
  const domes = new THREE.InstancedMesh(new THREE.SphereGeometry(m(2.1), 10, 5, 0, Math.PI * 2, 0, Math.PI / 2), dome, n);
  const mx = new THREE.Matrix4();
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    domes.setMatrixAt(i, mx.makeTranslation(Math.cos(a) * m(ARC_IN + 3), m(ARC_H), Math.sin(a) * m(ARC_IN + 3)));
  }
  g.add(domes);

  // the grey Mataf expansion: the lower level, the upper one stepped back (its open gallery dark), the roof on top
  g.add(ring(EXP[0].r0, EXP_OUT, 0, EXP[0].h, [greyTop, greyShade]));
  g.add(ring(EXP[1].r0, EXP_OUT, EXP[0].h, EXP[1].h, [roof, grey]));

  const topAt = (x: number, z: number) => {
    const r = Math.hypot(x, z) / S;
    return m(r < ARC_IN ? 0 : r < ARC_OUT ? ARC_H : r > EXP_OUT ? 0 : [...EXP].reverse().find((l) => r >= l.r0)?.h ?? 0);
  };
  const extent: THREE.Vector3[] = [];
  for (let i = 0; i < 32; i++) {
    const a = (i / 32) * Math.PI * 2, x = Math.cos(a), z = Math.sin(a);
    extent.push(new THREE.Vector3(x * m(PLINTH_R + 1), -m(1), z * m(PLINTH_R + 1)),
      new THREE.Vector3(x * m(EXP_OUT), m(EXP[1].h), z * m(EXP_OUT)));
  }
  return { group: g, topAt, floors: [0, LEVEL.first, LEVEL.band, LEVEL.roof].map(m), extent };
}
