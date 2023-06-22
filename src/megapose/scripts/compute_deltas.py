import torch

def transformation_differences(matrices1, matrices2):
    """
    Compute the translation and rotational differences between corresponding transformation matrices.
    
    Parameters:
        matrices1 (torch.Tensor): A tensor of shape (N, 4, 4) containing N transformation matrices.
        matrices2 (torch.Tensor): A tensor of shape (N, 4, 4) containing N transformation matrices.
    
    Returns:
        translation_diff (torch.Tensor): A tensor of shape (N, 3) containing translation differences.
        rotation_diff_euler (torch.Tensor): A tensor of shape (N, 3) containing rotational differences in Euler angles.
    """
    # Check input shapes
    assert matrices1.shape == matrices2.shape == (matrices1.shape[0], 4, 4), "Input tensors must have shape (N, 4, 4)"

    # Extract translation components
    translations1 = matrices1[:, :3, 3]
    translations2 = matrices2[:, :3, 3]
    
    # Compute translation differences
    translation_diff = translations1 - translations2
    
    # Extract rotation matrices
    rotations1 = matrices1[:, :3, :3]
    rotations2 = matrices2[:, :3, :3]

    # Compute relative rotation matrices
    rel_rotations = torch.bmm(rotations1, rotations2.transpose(1, 2))

    # Convert relative rotation matrices to Euler angles
    rotation_diff_euler = torch.zeros((matrices1.shape[0], 3))
    for i in range(matrices1.shape[0]):
        rotation_diff_euler[i] = matrix_to_euler(rel_rotations[i])
    
    return translation_diff, rotation_diff_euler


def matrix_to_euler(matrix):
    """
    Convert a rotation matrix to Euler angles.
    
    Parameters:
        matrix (torch.Tensor): A tensor of shape (3, 3) representing a rotation matrix.

    Returns:
        torch.Tensor: A tensor of shape (3,) representing the Euler angles (roll, pitch, yaw).
    """
    pitch = torch.asin(-matrix[2, 0])
    if torch.abs(pitch - torch.tensor(3.141592) / 2) < 1e-3:
        roll = 0
        yaw = torch.atan2(matrix[1, 2] - matrix[0, 1], matrix[1, 1] + matrix[0, 2])
    else:
        roll = torch.atan2(matrix[2, 1], matrix[2, 2])
        yaw = torch.atan2(matrix[1, 0], matrix[0, 0])
    return torch.tensor([roll, pitch, yaw])
    

def interpolate_translation(TCO_input, TCO_output, alpha):
    """
    Interpolate the translation components of transformation matrices.
    
    Parameters:
        TCO_input (torch.Tensor): A tensor of shape (N, 4, 4) containing N transformation matrices.
        TCO_output (torch.Tensor): A tensor of shape (N, 4, 4) containing N transformation matrices.
        alpha (float): Interpolation factor, 0 <= alpha <= 1. 0 returns TCO_input and 1 returns TCO_output.
        
    Returns:
        torch.Tensor: A tensor of shape (N, 4, 4) containing N interpolated transformation matrices.
    """
    # Validate input shapes
    assert TCO_input.shape == TCO_output.shape == (TCO_input.shape[0], 4, 4), "Input tensors must have shape (N, 4, 4)"
    assert 0 <= alpha <= 1, "alpha must be between 0 and 1"

    # Extract translation components
    translations_input = TCO_input[:, :3, 3]
    translations_output = TCO_output[:, :3, 3]
    
    # Interpolate translations
    interpolated_translations = (1 - alpha) * translations_input + alpha * translations_output
    
    # Construct interpolated transformation matrices
    interpolated_matrices = TCO_output.clone()
    interpolated_matrices[:, :3, 3] = interpolated_translations
    
    return interpolated_matrices