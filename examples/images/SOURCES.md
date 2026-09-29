# Sources of the test images

`scripts/evaluate_inpainting.py` cuts holes into these images and compares the fills
with the original content. They are taken from the sample data of scikit-image 0.26
(`skimage.data`), which documents their sources and licenses; `brick`, `grass` and
`gravel` are grayscale.

| File | Content | Source | License |
| --- | --- | --- | --- |
| `brick.png` | brick wall | [CC0Textures, Bricks25](https://cc0textures.com/view.php?tex=Bricks25) (now ambientCG), transformed, cropped and scaled by scikit-image | CC0 1.0 |
| `grass.png` | grass | [DeviantArt, linolafett, Grass 01](https://www.deviantart.com/linolafett/art/Grass-01-434853879), cropped by scikit-image | CC0 1.0 |
| `gravel.png` | gravel | [CC0Textures, Gravel04](https://cc0textures.com/view.php?tex=Gravel04) (now ambientCG), scaled and cropped by scikit-image | CC0 1.0 |
| `coffee.png` | coffee cup | photograph by Rachel Michetti, courtesy of Pikolo Espresso Bar | CC0 1.0 |
| `chelsea.png` | Chelsea the cat | photograph by Stefan van der Walt | CC0 1.0 |
