import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF, Center } from "@react-three/drei";
import * as THREE from "three";
import { ORANGE } from "../../constants/theme";

// 3D Porsche Model Component
export function PorscheModel() {
  const groupRef = useRef<THREE.Group>(null);
  const { scene } = useGLTF("/models/porsche911.glb");

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.position.y =
        Math.sin(state.clock.elapsedTime * 1.5) * 0.02;
    }
  });

  return (
    <group ref={groupRef}>
      <Center>
        <primitive
          object={scene.clone()}
          scale={0.8}
          rotation={[0, Math.PI / 5, 0]}
        />
      </Center>
    </group>
  );
}

useGLTF.preload("/models/porsche911.glb");

// Loading spinner for 3D model
export function ModelLoader() {
  const meshRef = useRef<THREE.Mesh>(null);
  useFrame((_, delta) => {
    if (meshRef.current) meshRef.current.rotation.y += delta * 2;
  });
  return (
    <mesh ref={meshRef}>
      <torusGeometry args={[1, 0.2, 16, 32]} />
      <meshStandardMaterial color={ORANGE} wireframe />
    </mesh>
  );
}
