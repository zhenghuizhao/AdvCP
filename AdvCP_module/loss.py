import torch
import torch.nn as nn
import torch.nn.functional as F
import random

###　pred_loss ###
def change_pred_loss_cam(pred, label, diff_cam_label=None, ignore_index=255, hard_sample_weight=10.0):
    # Create a weight map with all ones
    weight_map = torch.ones_like(label, dtype=torch.float32)

    # Alternatively, use diff_cam_label_values as weights
    if diff_cam_label is not None:
        weight_map = diff_cam_label * hard_sample_weight

    # Calculate loss for background
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), reduction='none',  ignore_index=ignore_index)

    # Apply the weights to the background loss
    bg_loss = bg_loss * weight_map

    # Calculate loss for foreground
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), reduction='none')

    # Apply the weights to the foreground loss
    fg_loss = fg_loss * weight_map

    # Combine the losses, while still treating them as separate channels
    final_loss = (bg_loss + fg_loss) * 0.5

    # Average the loss over all pixels
    mask = weight_map != ignore_index
    final_loss = final_loss[mask].mean()

    return final_loss

def change_pred_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    final_loss = (bg_loss + fg_loss) * 0.5
    # final_loss = final_loss.mean()
    return final_loss

def change_pred_loss_um(pred, label, diff_cam_label, ignore_index=255, hard_sample_weight=2.0):
    # Create a weight map with all ones
    weight_map = torch.ones_like(label, dtype=torch.float32)

    # Increase the weight of the hard samples identified by xor_label
    weight_map[diff_cam_label] = hard_sample_weight

    # Calculate loss for background
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), reduction='none', ignore_index=ignore_index)

    # Apply the weights to the background loss
    bg_loss = bg_loss * weight_map

    # Calculate loss for foreground
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), reduction='none', ignore_index=ignore_index)

    # Apply the weights to the foreground loss
    fg_loss = fg_loss * weight_map
    # Combine the losses, while still treating them as separate channels
    final_loss = (bg_loss + fg_loss) * 0.5

    # Average the loss over all non-ignored pixels
    mask = weight_map != ignore_index
    final_loss = final_loss[mask].mean()

    return final_loss




### center loss ###
def update_centers(pred, label, centers, alpha=0.9, ignore_index=255):
    """
    Update the centers based on the current batch, ignoring pixels with the ignore_index.

    Args:
    pred (torch.Tensor): Prediction from the model for the current batch. Shape: [batch_size, num_classes, height, width].
    label (torch.Tensor): Ground truth labels for the current batch. Shape: [batch_size, height, width].
    centers (torch.Tensor): Current centers for each class. Shape: [num_classes, feature_dim].
    alpha (float): Learning rate for updating centers.
    ignore_index (int): Label value to ignore when updating centers.

    Returns:
    torch.Tensor: Updated centers.
    """
    with torch.no_grad():  # Ensure no gradient is computed
        batch_size, num_classes, height, width = pred.shape
        feature_dim = num_classes  # Assuming the prediction is class scores
        pred = pred.permute(0, 2, 3, 1).reshape(-1, feature_dim)
        label = label.reshape(-1)

        for i in range(num_classes):
            # Exclude pixels with the ignore_index
            class_mask = (label == i) & (label != ignore_index)
            class_features = pred[class_mask]

            if class_features.size(0) > 0:
                class_center = class_features.mean(0)
                centers[i] = (1-alpha) * class_center + alpha * centers[i]

    return centers


def acc_center_loss(pred, diff_cam_label, centers, lambda_center=1.0, ignore_index=255, hard_mining_weight=10.0):
    """
        Calculate the center loss for diff_cam_label pixels based on dataset centers, ignoring pixels with ignore_index.
        Pixels with diff_cam_label are considered as hard samples and weighted more heavily in the loss.

        Args:
        pred (torch.Tensor): Prediction from the model. Shape: [batch_size, num_classes, height, width].
        diff_cam_label (torch.Tensor): Boolean tensor indicating pixels to consider as hard samples. Shape: [batch_size, 1, height, width].
        centers (torch.Tensor): Centers for each class.
        lambda_center (float): Scaling factor for the center loss.
        ignore_index (int): Label value to ignore when calculating the loss.
        hard_mining_weight (float): Weighting factor for hard samples in the loss.

        Returns:
        torch.Tensor: The calculated center loss.
        """
    batch_size, num_classes, height, width = pred.shape
    feature_dim = num_classes  # Assuming the prediction is class scores
    pred = pred.permute(0, 2, 3, 1).reshape(-1, feature_dim)
    diff_cam_label = diff_cam_label.view(-1)

    # Exclude pixels with the ignore_index
    valid_mask = diff_cam_label != ignore_index
    selected_pred = pred[valid_mask]
    selected_diff_cam_label = diff_cam_label[valid_mask]

    if selected_pred.nelement() == 0:
        return torch.tensor(0.0, device=pred.device)

    selected_labels = selected_pred.argmax(1)
    distances = torch.norm(selected_pred - centers[selected_labels], dim=1)

    # Apply hard mining weight to diff_cam_label samples
    weights = torch.ones_like(distances)
    weights[selected_diff_cam_label] *= hard_mining_weight

    weighted_center_loss = (distances * weights).mean() * lambda_center

    return weighted_center_loss


def modified_center_loss(pred, diff_cam_label, centers, margin=0.5, ignore_index=255):
    """
    Calculate a soft margin dual center loss for diff_cam_label pixels based on dataset centers,
    ignoring pixels with ignore_index.
    This loss function encourages diff_cam_label pixels to be closer to their class center than to other class centers by a margin.

    Args:
    pred (torch.Tensor): Prediction from the model. Shape: [batch_size, num_classes, height, width].
    diff_cam_label (torch.Tensor): Boolean tensor indicating pixels to consider. Shape: [batch_size, 1, height, width].
    centers (torch.Tensor): Centers for each class.
    margin (float): Soft margin to encourage separability.

    Returns:
    torch.Tensor: The calculated soft margin dual center loss.
    """
    batch_size, num_classes, height, width = pred.shape
    feature_dim = num_classes  # Assuming the prediction is class scores
    pred = pred.permute(0, 2, 3, 1).reshape(-1, feature_dim)
    diff_cam_label = diff_cam_label.view(-1)

    # Exclude pixels with the ignore_index
    valid_mask = diff_cam_label != ignore_index
    selected_pred = pred[valid_mask]

    if selected_pred.nelement() == 0:
        return torch.tensor(0.0, device=pred.device)

    selected_labels = selected_pred.argmax(1)

    # Calculate distances to own center
    distances_to_own_center = torch.norm(selected_pred - centers[selected_labels], dim=1)

    # Calculate distances to other centers and apply soft margin
    loss = 0.0
    for i in range(num_classes):
        mask = (selected_labels != i) & valid_mask
        if torch.any(mask):
            distances_to_other_centers = torch.norm(selected_pred[mask] - centers[i], dim=1)
            # Apply soft margin
            loss += F.relu(distances_to_own_center[mask] - distances_to_other_centers + margin).mean()

    loss /= num_classes - 1

    return loss

def calculate_feature_centers(pred, diff_cam_label, num_classes, ignore_index=255):
    with torch.no_grad():
        feature_dim = num_classes
        device = pred.device
        pred = pred.permute(0, 2, 3, 1).reshape(-1, feature_dim)
        diff_cam_label = diff_cam_label.view(-1)

        # initialize
        with_diff_accumulated_centers = torch.zeros(num_classes, feature_dim, device=device)
        without_diff_accumulated_centers = torch.zeros(num_classes, feature_dim, device=device)
        with_diff_pixel_counts = torch.zeros(num_classes, device=device)
        without_diff_pixel_counts = torch.zeros(num_classes, device=device)

        for i in range(num_classes):
            class_mask = pred.argmax(1) == i
            valid_mask = diff_cam_label != ignore_index

            with_diff_mask = class_mask & valid_mask & diff_cam_label
            without_diff_mask = class_mask & valid_mask & ~diff_cam_label

            # update
            if with_diff_mask.any():
                with_diff_accumulated_centers[i], with_diff_pixel_counts[i] = update_consistency_center(pred, with_diff_mask, with_diff_accumulated_centers[i], with_diff_pixel_counts[i])


            if without_diff_mask.any():
                without_diff_accumulated_centers[i], without_diff_pixel_counts[i] = update_consistency_center(pred, without_diff_mask, without_diff_accumulated_centers[i], without_diff_pixel_counts[i])

        return with_diff_accumulated_centers, without_diff_accumulated_centers

def update_consistency_center(pred, mask, accumulated_center, pixel_count):
    new_center = pred[mask].mean(0)
    total_pixels = pixel_count + mask.sum()
    updated_center = (accumulated_center * pixel_count + new_center * mask.sum()) / total_pixels
    return updated_center, total_pixels



def calculate_feature_centers(pred, diff_cam_label, num_classes, ignore_index=255):
    """
    Calculate feature centers for pixels with and without diff_cam_label, ignoring pixels with ignore_index.

    Args:
    pred (torch.Tensor): Prediction from the model. Shape: [batch_size, num_classes, height, width].
    diff_cam_label (torch.Tensor): Boolean tensor indicating pixels to consider. Shape: [batch_size, 1, height, width].
    num_classes (int): Number of classes.
    ignore_index (int): Label value to ignore when calculating centers.

    Returns:
    torch.Tensor: Centers for pixels with diff_cam_label.
    torch.Tensor: Centers for pixels without diff_cam_label.
    """
    batch_size, _, height, width = pred.shape
    feature_dim = num_classes  # Assuming the prediction is class scores
    pred = pred.permute(0, 2, 3, 1).reshape(-1, feature_dim)
    diff_cam_label = diff_cam_label.view(-1)

    with_diff_centers = torch.zeros(num_classes, feature_dim, device=pred.device)
    without_diff_centers = torch.zeros(num_classes, feature_dim, device=pred.device)

    for i in range(num_classes):
        class_mask = (pred.argmax(1) == i) & (diff_cam_label != ignore_index)
        with_diff_mask = class_mask & diff_cam_label
        without_diff_mask = class_mask & ~diff_cam_label

        if with_diff_mask.any():
            with_diff_centers[i] = pred[with_diff_mask].mean(0)

        if without_diff_mask.any():
            without_diff_centers[i] = pred[without_diff_mask].mean(0)

    return with_diff_centers, without_diff_centers



### consistency loss ###
def consistency_loss(with_diff_centers, without_diff_centers, hard_mining_weight=1.0):

    """
    Calculate the loss to minimize the distance between two sets of centers.

    Args:
    with_diff_centers (torch.Tensor): Centers for pixels with diff_cam_label.
    without_diff_centers (torch.Tensor): Centers for pixels without diff_cam_label.

    Returns:
    torch.Tensor: The calculated loss.
    """
    loss = hard_mining_weight * F.mse_loss(with_diff_centers, without_diff_centers)
    return loss



### contrastive loss ###
def contrastive_loss(pred, diff_cam_label, centers, margin=0.5, sample_size=50000, hard_mining_weight=2.0):
    """
    Calculate a contrastive loss for randomly selected pixel samples,
    ensuring that diff_cam_label pixels are included and weighted more if they are hard samples.

    Args:
    pred (torch.Tensor): Prediction from the model. Shape: [batch_size, num_classes, height, width].
    diff_cam_label (torch.Tensor): Tensor indicating important (hard) pixels. Shape: [batch_size, 1, height, width].
    centers (torch.Tensor): Centers for each class.
    margin (float): Margin to encourage separability.
    sample_size (int): Number of samples to select for the loss calculation.
    hard_mining_weight (float): Weighting factor for hard samples in the loss.

    Returns:
    torch.Tensor: The calculated contrastive loss.
    """
    batch_size, num_classes, height, width = pred.shape
    feature_dim = num_classes  # Assuming the prediction is class scores
    pred = pred.permute(0, 2, 3, 1).reshape(-1, feature_dim)
    diff_cam_label = diff_cam_label.view(-1)

    # Select samples where diff_cam_label is important (hard samples)
    important_indices = torch.nonzero(diff_cam_label).view(-1)
    important_samples = random.sample(important_indices.tolist(), min(sample_size, len(important_indices)))

    # Select random samples from the rest
    other_indices = torch.nonzero(diff_cam_label == 0).view(-1)
    other_samples = random.sample(other_indices.tolist(), min(sample_size, len(other_indices)))

    # Combine samples and create weights
    selected_samples = important_samples + other_samples
    selected_pred = pred[selected_samples]
    selected_labels = selected_pred.argmax(1)

    device = pred.device  # 获取pred张量所在的设备
    sample_weights = torch.ones(len(selected_samples), device=device)
    sample_weights[:len(important_samples)] *= hard_mining_weight

    # Calculate contrastive loss
    loss = 0.0
    for i in range(num_classes):
        class_mask = selected_labels == i
        class_features = selected_pred[class_mask]
        class_weights = sample_weights[class_mask]

        if class_features.size(0) > 0:
            distances_to_own_center = torch.norm(class_features - centers[i], dim=1)
            intra_class_loss = (distances_to_own_center * class_weights).mean()

            # Inter-class (to other classes)
            inter_class_loss = 0.0
            for j in range(num_classes):
                if j != i:
                    distances_to_other_centers = torch.norm(class_features - centers[j], dim=1)
                    inter_class_loss += (F.relu(margin - distances_to_other_centers) * class_weights).mean()

            loss += intra_class_loss + F.relu(margin - inter_class_loss)

    return loss / num_classes
