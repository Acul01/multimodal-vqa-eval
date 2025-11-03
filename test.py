# test_load_data.py
from utils.load_data import load_vqax, load_actx, load_esnlive, load_vcr

images_root = "/netscratch/lrippe/project_scripts/llava/images"
nle_root    = "/netscratch/lrippe/project_scripts/llava/nle_data"

print("\n=== VQA-X ===")
vqax = load_vqax(images_root, f"{nle_root}/VQA-X", split="val", require_image=False)
print("Samples:", len(vqax))
if vqax:
    s = vqax[0]
    print("  Q:", s.question)
    print("  A:", s.answer)
    print("  Expl:", s.explanation)
    print("  Image (expected path):", s.image_path)

print("\n=== ACT-X ===")
actx = load_actx(images_root, f"{nle_root}/ACT-X", split="test", require_image=False)  # val→test
print("Samples:", len(actx))
if actx:
    s = actx[0]
    print("  Label:", s.label)
    print("  Expl:", s.explanation)
    print("  Image (expected path):", s.image_path)

print("\n=== e-SNLI-VE ===")
esv = load_esnlive(images_root, f"{nle_root}/eSNLI-VE", split="test", require_image=False)  # val→test
print("Samples:", len(esv))
if esv:
    s = esv[0]
    print("  Hyp:", s.hypothesis)
    print("  Label:", s.label)
    print("  Expl:", s.explanation)
    print("  Image (expected path):", s.image_path)

print("\n=== VCR ===")
vcr = load_vcr(images_root, f"{nle_root}/VCR", split="val", require_image=False)
print("Samples:", len(vcr))
if vcr:
    s = vcr[0]
    print("  Q:", s.question)
    print("  A:", s.answer)
    print("  Expl:", s.explanation)
    print("  Image (expected path):", s.image_path)