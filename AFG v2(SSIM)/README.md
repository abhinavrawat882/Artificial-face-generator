# ABOUT

This uses new SSIM loss. This is loss is a great improvement over MSE or PSNR .

# SSIM function

SSIM returns values beetween -1 to 1. -1 if images are totally different and 1 for similar images.

So you have to negate the returned value from the function .. to use it as loss function.
