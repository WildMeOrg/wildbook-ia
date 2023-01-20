This shows example code for applying some of the pieces of WBIA to images to
form annotations, filter them, and run Hotspotter.

1. Localizer
2. Labeler
3. CA regions
4. Hotspotter

CA regions is optional in the current configuration.

Other many other operations may be applied to refine and filter
annotations.

1. Bounding box regressions to form census annotation regions
2. Removing annotations with extreme bounding box aspect ratios
3. Removing annotations that are too small
4. Removing annotations that have too low contrast.

Most of these would be unnecessary with larger volumes
of training data, but we don't have that luxury.
