// 球面メッシュの細かさを設定
SetFactory("OpenCASCADE");

// 球を定義
Sphere(1) = {0, 3.25, 3.25, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(2) = {0, 1, 3.25, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(3) = {0, -1.25, 3.25, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(4) = {0, -3.5, 3.25, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(5) = {0, 3.25, 1, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(6) = {0, 1, 1, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(7) = {0, -1.25, 1, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(8) = {0, -3.5, 1, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(9) = {0, 3.25, -1.25, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(10) = {0, 1, -1.25, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(11) = {0, -1.25, -1.25, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(12) = {0, -3.5, -1.25, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(13) = {0, 3.25, -3.5, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(14) = {0, 1, -3.5, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(15) = {0, -1.25, -3.5, 1, -Pi/2, Pi/2, 2*Pi};
Sphere(16) = {0, -3.5, -3.5, 1, -Pi/2, Pi/2, 2*Pi};
Mesh 2;
Mesh.SurfaceFaces = 1;
ReverseMesh Surface {:};
