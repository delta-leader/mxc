// 球面メッシュの細かさを設定
SetFactory("OpenCASCADE");

// 球を定義
Sphere(1) = {1, 1, 1, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(2) = {1, -1.25, 1, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(3) = {1, 1, -1.25, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(4) = {1, -1.25, -1.25, 1, -Pi/2, Pi/2, 2*Pi};
Mesh 2;
Mesh.SurfaceFaces = 1;
ReverseMesh Surface {:};
