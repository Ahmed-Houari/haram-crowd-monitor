// The Kaaba, Hijr Ismail and Maqam Ibrahim for the 3D twin, built from published measurements (sources in
// docs/design/README.md §7). Scale ~0.116 scene units per metre (the Mataf floor radius 10.6 = ~91 m).
// Local axes: +z = the north-east wall with the door, +x = the north-west wall facing Hijr Ismail,
// the Black Stone sits on the east corner (-x, +z).
import * as THREE from "three";
import { doorMaps, hizamMaps, kiswaMaps, marbleMap } from "./twinTextures";

const S = 0.116;
const m = (metres: number) => metres * S;
export const KAABA_ROT = 0.32; // the Kaaba frame in the scene (the architecture and the crowd field share it)
export const KAABA_W = 12.86, KAABA_D = 11.03; // m: the door wall (along x) and the Hijr wall (along z)
export const MAQAM = { x: -KAABA_W / 2 + 3.8, z: KAABA_D / 2 + 11 }; // m, Kaaba frame: ~11 m in front of the door

const shadow = <T extends THREE.Object3D>(o: T, cast = true, receive = true): T => {
  o.traverse((x) => {
    x.castShadow = cast;
    x.receiveShadow = receive;
  });
  return o;
};

export function buildKaaba(): THREE.Group {
  const g = new THREE.Group();
  g.rotation.y = KAABA_ROT;
  const W = m(KAABA_W), D = m(KAABA_D), H = m(13.1); // door wall x Hijr wall x height

  const gold = new THREE.MeshStandardMaterial({ color: 0xd6b36a, metalness: 1, roughness: 0.32 });
  const marble = new THREE.MeshStandardMaterial({ map: marbleMap(512, [2, 2]), roughness: 0.38, color: 0xf2efe8 });

  // Kiswa: black silk with a woven pattern; the roof is not covered. No text anywhere on the model.
  const k = kiswaMaps();
  const cloth = new THREE.MeshPhysicalMaterial({ map: k.map, roughnessMap: k.rm, roughness: 1, metalness: 0,
    sheen: 0.7, sheenColor: new THREE.Color(0x3c3c44), sheenRoughness: 0.4 });
  const roof = new THREE.MeshStandardMaterial({ color: 0x55534e, roughness: 0.85 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(W, H, D), [cloth, cloth, roof, roof, cloth, cloth]);
  body.position.y = H / 2;
  g.add(shadow(body));

  // Hizam: gold belt in the upper third of the walls, 0.95 m tall, a hair proud of the cloth (abstract ornament)
  const belts = { nw: hizamMaps(1), se: hizamMaps(2), ne: hizamMaps(3), sw: hizamMaps(4) };
  const beltMat = (b: { map: THREE.Texture; rm: THREE.Texture }) =>
    new THREE.MeshStandardMaterial({ map: b.map, roughnessMap: b.rm, metalnessMap: b.rm, roughness: 1, metalness: 1 });
  const belt = new THREE.Mesh(new THREE.BoxGeometry(W + m(0.1), m(0.95), D + m(0.1)),
    [beltMat(belts.nw), beltMat(belts.se), gold, gold, beltMat(belts.ne), beltMat(belts.sw)]);
  belt.position.y = H * 0.77;
  g.add(shadow(belt));

  // Door, on the north-east wall near the Black Stone corner: the gold door (3.1 x 1.9 m, foot 2.13 m above the
  // floor) under its gold-embroidered curtain (Sitara, ~5.75 x 3.5 m), which hangs from the door's foot upwards
  const doorX = -W / 2 + m(2.95);
  const dm = doorMaps();
  // part-metal so it reads as bright gold under the dim night light (full metal only mirrors the dark sky)
  const sitaraMat = new THREE.MeshStandardMaterial({ map: dm.map, roughnessMap: dm.rm, roughness: 1, metalness: 0.55 });
  const sitara = new THREE.Mesh(new THREE.BoxGeometry(m(3.5), m(5.75), m(0.08)), [gold, gold, gold, gold, sitaraMat, gold]);
  sitara.position.set(doorX, m(2.13 + 5.75 / 2), D / 2 + m(0.04));
  g.add(shadow(sitara));

  // Black Stone: set into the east corner at ~1.5 m, in its silver frame, which wraps around the corner flush
  // with both walls (an upright oval split by the corner edge, with a round opening for the stone)
  const silver = new THREE.MeshStandardMaterial({ color: 0xeef1f3, metalness: 0.65, roughness: 0.28, side: THREE.DoubleSide });
  const fa = m(0.32), fb = m(0.45), fr = m(0.16), steps = 20;
  const half = new THREE.Shape(); // half-oval on one wall, the stone's half-circle cut out at the corner (u = 0)
  half.moveTo(0, fb);
  for (let i = 1; i <= steps; i++) {
    const t = Math.PI / 2 - (Math.PI * i) / steps;
    half.lineTo(fa * Math.cos(t), fb * Math.sin(t));
  }
  half.lineTo(0, -fr);
  for (let i = 1; i < steps; i++) {
    const t = -Math.PI / 2 + (Math.PI * i) / steps;
    half.lineTo(fr * Math.cos(t), fr * Math.sin(t));
  }
  half.lineTo(0, fr);
  half.closePath();
  const plate = (mirror: boolean) => {
    const geo = new THREE.ExtrudeGeometry(half, { depth: m(0.025), bevelEnabled: false });
    if (mirror) geo.scale(-1, 1, 1);
    return new THREE.Mesh(geo, silver);
  };
  const stoneY = m(1.5);
  const onDoorWall = plate(false); // north-east wall: runs +x from the corner, faces +z
  onDoorWall.position.set(-W / 2, stoneY, D / 2 + m(0.004));
  const onSideWall = plate(true); // south-east wall: runs -z from the corner, faces -x
  onSideWall.rotation.y = -Math.PI / 2;
  onSideWall.position.set(-W / 2 - m(0.004), stoneY, D / 2);
  const stone = new THREE.Mesh(new THREE.SphereGeometry(fr, 24, 16), new THREE.MeshStandardMaterial({ color: 0x1c1514, roughness: 0.5 }));
  stone.position.set(-W / 2 + m(0.08), stoneY, D / 2 - m(0.08)); // mostly inside the corner, a cap shows
  g.add(shadow(onDoorWall), shadow(onSideWall), shadow(stone));

  // Shadharwan: the sloping marble base, on three walls (not on the Hijr side)
  const wedge = new THREE.Shape([new THREE.Vector2(0, 0), new THREE.Vector2(m(0.4), 0), new THREE.Vector2(m(0.4), m(0.06)),
    new THREE.Vector2(0, m(0.3))]);
  const base = (len: number, rotY: number, x: number, z: number) => {
    const geo = new THREE.ExtrudeGeometry(wedge, { depth: len, bevelEnabled: false });
    geo.translate(0, 0, -len / 2);
    const mesh = new THREE.Mesh(geo, marble);
    mesh.rotation.y = rotY;
    mesh.position.set(x, 0, z);
    return shadow(mesh, false, true);
  };
  g.add(base(W + m(0.8), -Math.PI / 2, 0, D / 2), base(W + m(0.8), Math.PI / 2, 0, -D / 2), base(D, Math.PI, -W / 2, 0));

  // Hijr Ismail: low semicircular white marble wall off the north-west wall, 1.32 m high, 0.9 m thick,
  // 8.44 m from the Kaaba at its middle, open at both ends (~2.3 m from the wall)
  const a = m(4.95), b = m(8.44), th = m(0.9), h = m(1.32), A = (74 * Math.PI) / 180, n = 48;
  const arc = new THREE.Shape();
  for (let i = 0; i <= n; i++) {
    const t = -A + (2 * A * i) / n;
    if (i === 0) arc.moveTo(b * Math.cos(t), a * Math.sin(t));
    else arc.lineTo(b * Math.cos(t), a * Math.sin(t));
  }
  for (let i = n; i >= 0; i--) {
    const t = -A + (2 * A * i) / n;
    arc.lineTo((b + th) * Math.cos(t), (a + th) * Math.sin(t));
  }
  const hijrGeo = new THREE.ExtrudeGeometry(arc, { depth: h, bevelEnabled: true, bevelThickness: m(0.06), bevelSize: m(0.06), bevelSegments: 2 });
  hijrGeo.rotateX(-Math.PI / 2); // shape (out, along) -> floor (x, -z), extruded up
  const hijr = new THREE.Mesh(hijrGeo, marble);
  hijr.position.x = W / 2;
  g.add(shadow(hijr));

  // Maqam Ibrahim, ~11 m in front of the door: marble base, gilded brass cage around a crystal cover
  // (0.8 m across, 1.3 m high), small dome with a crescent finial. Drawn 1.5x so it reads at dashboard size.
  const maqam = new THREE.Group();
  maqam.position.set(m(MAQAM.x), 0, m(MAQAM.z));
  maqam.scale.setScalar(1.5);
  const plinth = new THREE.Mesh(new THREE.CylinderGeometry(m(0.62), m(0.66), m(0.3), 8), marble);
  plinth.position.y = m(0.15);
  const glass = new THREE.Mesh(new THREE.CylinderGeometry(m(0.36), m(0.36), m(0.85), 24, 1, true),
    new THREE.MeshPhysicalMaterial({ color: 0xcfe6ea, metalness: 0, roughness: 0.05, transmission: 0, transparent: true, opacity: 0.28, side: THREE.DoubleSide }));
  glass.position.y = m(0.3 + 0.425);
  maqam.add(plinth, glass);
  for (let i = 0; i < 8; i++) { // cage bars at the eight corners
    const q = (i / 8) * Math.PI * 2;
    const bar = new THREE.Mesh(new THREE.BoxGeometry(m(0.05), m(0.85), m(0.05)), gold);
    bar.position.set(Math.cos(q) * m(0.4), m(0.3 + 0.425), Math.sin(q) * m(0.4));
    maqam.add(bar);
  }
  for (const y of [0.32, 0.72, 1.13]) { // rings
    const ring = new THREE.Mesh(new THREE.TorusGeometry(m(0.41), m(0.025), 6, 32), gold);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = m(y);
    maqam.add(ring);
  }
  const dome = new THREE.Mesh(new THREE.LatheGeometry([0, 0.12, 0.3, 0.42, 0.45, 0.47].map((r, i, all) =>
    new THREE.Vector2(m(r), m(0.32 * (1 - i / (all.length - 1))))).reverse(), 24), gold);
  dome.position.y = m(1.15);
  const crescent = new THREE.Mesh(new THREE.TorusGeometry(m(0.07), m(0.015), 6, 16, Math.PI * 1.5), gold);
  crescent.position.y = m(1.15 + 0.32 + 0.12);
  crescent.rotation.z = Math.PI * 0.75;
  maqam.add(dome, crescent);
  g.add(shadow(maqam));

  return g;
}
