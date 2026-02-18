Task: Procedural Shadow Synthesis via Lighting Re-Projection.

Scene Parameters: Image A (Donor) serves as the lighting and shadow master. Image B (Recipient) contains the target subject.

Directive: Reconstruct the unattached shadows for the Recipient subject by adopting the identical lighting architecture present in the Donor image for them. You must calculate the unattached shadows based on a light source with the exact spatial coordinates, distance, and radiant intensity as observed in the Donor.

Execution Details:

Geometric Projection: Map the unattached shadows from the Donor to the Recipient’s specific contact points, ensuring the angular projection matches the Donor's light vector as if it were now present in the Recipient image. Unrealistically the light creating the unattached shadows doews nothting to alter the appearance fo the Recipient subject.

Photometric Fidelity: Replicate the Donor’s penumbra profile (edge softness) and ambient occlusion (depth of darkness at the base) for the unattached shadows in the Recipient.

Integrity: The Recipient subject’s materials and textures must remain untouched; only the floor-plane unattached shadows and contact-point occlusions are to be synthesized. Mask the DOnor subject and its cast shadows to protect them from any further changes make fill all other areas in the Recipient pure white.

Finally: Lock down the composit as is. no pixels may change until persmission is given. On a side canvas we construct another image called last_layer. To create it we remove the background from a copy of the original submitted Recipient. Then we carefully line it up with the subject in main image we constructed and locked down. We wont be able to change an pixels in the composit but we can cover the existing subject with an ideantical one except for the robust colors who ignores the lighting sources amd maintains its original luminocity and colors from the lighting in the Recipient image, straight onto our existing composit without anything in the environment altering its appearance. We want a perfect copy of the original subject placed over the one that is there because we like the colors better than the present version. At this point out final composite is made of the Recipient subject and the unattached shadows. Other than that it is (rgb 255,255,255 white).

Output: A composite where the Recipient subject's unattached shadows appear to be cast as if they were cast from the exact same lighting environment as the Donor. With a solid white background.