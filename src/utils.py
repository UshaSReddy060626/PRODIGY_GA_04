import os
import matplotlib
matplotlib.use("Agg")  # no display needed; just save PNGs to disk
import matplotlib.pyplot as plt
def generate_and_save_images(model, test_input, target, out_path):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(12, 4))
    display_list = [test_input[0], target[0], prediction[0]]
    titles = ["Input", "Ground Truth", "Predicted"]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        # rescale from [-1, 1] back to [0, 1] for display
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved sample comparison to: {out_path}")
