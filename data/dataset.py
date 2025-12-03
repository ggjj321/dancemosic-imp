import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class AISTDataset(Dataset):
    def __init__(self, data_dir, split='train', window_size=60, stride=20):
        """
        Args:
            data_dir (str): Path to the AIST++ dataset directory.
            split (str): 'train', 'val', or 'test'.
            window_size (int): Number of frames in each motion window.
            stride (int): Stride for sliding window.
        """
        self.data_dir = data_dir
        self.split = split
        self.window_size = window_size
        self.stride = stride
        
        self.motion_files = self._get_motion_files()
        self.data = self._load_data()

    def _get_motion_files(self):
        # Placeholder: Implement logic to list files based on split
        # Usually AIST++ has a split file or specific naming convention
        # For now, just list all .pkl files in the directory
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        # Simple split logic for demonstration
        if self.split == 'train':
            return files[:int(0.8 * len(files))]
        elif self.split == 'val':
            return files[int(0.8 * len(files)):]
        else:
            return files # Test uses all or specific subset

    def _load_data(self):
        data_chunks = []
        for filename in self.motion_files:
            file_path = os.path.join(self.data_dir, filename)
            try:
                with open(file_path, 'rb') as f:
                    motion_data = pickle.load(f)
                
                # Assuming motion_data is a dictionary or array with 'pose' or 'smpl_poses'
                # Adjust key based on actual data format
                if isinstance(motion_data, dict):
                    poses = motion_data.get('smpl_poses', motion_data.get('pose', None))
                else:
                    poses = motion_data # Assume array
                
                if poses is None:
                    continue

                # Slice into windows
                n_frames = poses.shape[0]
                for i in range(0, n_frames - self.window_size + 1, self.stride):
                    window = poses[i : i + self.window_size]
                    data_chunks.append(window)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return data_chunks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        motion = self.data[idx]
        # Convert to tensor
        motion = torch.from_numpy(motion).float()
        return motion

if __name__ == "__main__":
    # Test the dataset
    # Create a dummy directory and file for testing
    os.makedirs("dummy_data", exist_ok=True)
    dummy_motion = np.random.rand(100, 72) # 100 frames, 72 SMPL params
    with open("dummy_data/test_motion.pkl", "wb") as f:
        pickle.dump({'smpl_poses': dummy_motion}, f)

    dataset = AISTDataset("dummy_data", split='train')
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        print(f"Sample shape: {dataset[0].shape}")
    
    # Cleanup
    import shutil
    shutil.rmtree("dummy_data")
