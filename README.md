# YACPT

YACPT (Yet Another Cuda Path Tracer) is a raytracer demonstrating the usage of monte-carlo-integration to solve
the rendering equation resulting in global illumination. Some preprocessing is done on the CPU, the raytracing itself
happens on CUDA-enabled GPUs.

Currently supported features:
- QT for a (very) minimal GUI
- trianglemeshes and spheres
- a sceneparser (work in progress, currently pretty unstable)
- a bounding-volume-hierarchy to speed up ray-triangle intersection tests
- a pinhole camera
- diffuse, specular & emissive materials

features i am looking into:
- dielectric materials
- textures
- instancing
- motion blur
- isosurfaces
- direct lighting
- depth of field
