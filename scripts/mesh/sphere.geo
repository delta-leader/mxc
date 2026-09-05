// 球面メッシュの細かさを設定
SetFactory("OpenCASCADE");

// 球を定義
Sphere(1) = {0, 0, 0, 1, -Pi/2, Pi/2, 2*Pi};
Mesh 2;
Mesh.SurfaceFaces = 1;
ReverseMesh Surface {:};
