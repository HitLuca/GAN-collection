from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter


def plot_save_samples(generated_samples, rows, columns, resolution, channels, outputs_dir, epoch, titles=None):
    if titles is None:
        plt.subplots(rows, columns, figsize=(7, 7))
    else:
        plt.subplots(rows, columns, figsize=(10, 9))

    k = 1
    for i in range(rows):
        for j in range(columns):
            plt.subplot(rows, columns, k)
            if titles is not None:
                plt.title(titles[k - 1])
            if channels == 3:
                plt.imshow((generated_samples[k - 1].reshape(resolution, resolution, channels) + 1.0) / 2.0)
            elif channels == 1:
                plt.imshow((generated_samples[k - 1].reshape(resolution, resolution) + 1.0) / 2.0, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            k += 1
    plt.tight_layout()

    if titles is None:
        plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(outputs_dir + str(epoch) + '.png')
    plt.close()


def plot_save_latent_space(generated_samples, rows, columns, resolution, channels, outputs_dir, epoch, title=None):
    if title is not None:
        plt.subplots(rows, columns, figsize=(7, 7.5))
        plt.suptitle(title)
    else:
        plt.subplots(rows, columns, figsize=(7, 7))

    k = 1
    for i in range(rows):
        for j in range(columns):
            plt.subplot(rows, columns, k)
            if channels == 3:
                plt.imshow((generated_samples[k - 1].reshape(resolution, resolution, channels) + 1.0) / 2.0)
            elif channels == 1:
                plt.imshow((generated_samples[k - 1].reshape(resolution, resolution) + 1.0) / 2.0, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            k += 1
    if title is not None:
        plt.tight_layout(rect=[0, 0, 1, .95])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(outputs_dir + str(epoch) + '_latent_space' + '.png')
    plt.close()


def plot_save_losses(losses, labels, outputs_dir, name, sigma=3):
    plt.figure(figsize=(15, 4.5))
    for loss, label in zip(losses, labels):
        loss = gaussian_filter(loss, sigma=sigma)
        plt.plot(loss, label=label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputs_dir + name + '.png')
    plt.close()
