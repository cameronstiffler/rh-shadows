Task: Spatial Alignment and Shadow Isolation

Input Parameters:

Donor Image: High-resolution subject (furniture).

Recipient Image: Baseline image with a smaller, identical subject.

Phase 1: Geometric Synchronization and Superimposition

Spatial Scaling: Quantify the dimensions of the furniture in the Recipient image. Execute a proportional resize of the Donor subject to achieve absolute dimensional congruency with the Recipient subject, strictly maintaining the original aspect ratio of the subject 


Positional Overlay: Superimpose the resized Donor subject directly onto the Recipient image. The alignment must be surgically precise, ensuring the Donor subject perfectly masks the Recipient subject's coordinates and footprint. With the subjects in both images perfectly aligned cropping donor image edges to match Recipient image dimensions.

Phase 2: Shadow Extraction and High-Key Masking

Unattached Shadow Segmentation: In the Donor image, identify and isolate the existing cast unattached shadows by the subject onto the floor surface. Utilize a generous selection radius to capture the full periphery of the ambient occlusion and unattached shadows. Make sure to cafefully caputure the existing unattached shadows in the image who may be long, thin and fade gradually.

Luminance Transformation (Background): Apply a global mask to all pixels outside the identified shadow regions, forcing them to absolute white (RGB: 255, 255, 255).

Chrominance Stripping (Shadows): Convert the isolated shadow regions into a grayscale (achromatic) palette. Preserve the original luminance gradients and textures while removing all saturation.

Phase 3: Final Rendering Specifications

Output: An image where the grayscale shadows are the solitary visual artifacts against a pure white void. 255 white must occupy the space where the furniture once was
