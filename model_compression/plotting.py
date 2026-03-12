import os 
import matplotlib.pyplot as plt
import wandb 
from model_compression.utils import steeper_sigmoid, denorm_for_vis

def save_visualization(image_clean_tensor, image_prompt_tensor, pred_mask, teacher_mask, prompt_xy, step, IMAGE_OUTPUT_DIR):
    x, y = int(prompt_xy[0]), int(prompt_xy[1])

    img_clean = denorm_for_vis(image_clean_tensor)
    img_prompt = denorm_for_vis(image_prompt_tensor)

    pred = steeper_sigmoid(pred_mask[0]).squeeze().detach().cpu().numpy()
    teach = steeper_sigmoid(teacher_mask[0]).squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_clean)
    axes[0].scatter([x], [y], c="red", s=12)
    axes[0].set_title("Clean Image + Point")

    axes[1].imshow(img_prompt)
    axes[1].scatter([x], [y], c="red", s=12)
    axes[1].set_title("RGB w/ DoG Prompt")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Student Prediction")

    axes[3].imshow(teach, cmap="gray")
    axes[3].set_title("Teacher Mask (logits→sigmoid)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(IMAGE_OUTPUT_DIR, f"vis_step{step}.png")
    plt.savefig(save_path)
    plt.close()
    wandb.log({"visualization": wandb.Image(save_path)})