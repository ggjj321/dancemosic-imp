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
        
        if len(self.data) == 0:
            print(f"WARNING: Dataset ({self.split}) is empty. Check data_dir: {self.data_dir}")
            # We don't raise here to allow 'verify_pipeline' to run if it handles it, 
            # but DataLoader will fail if num_samples=0.

    def _get_motion_files(self):
        files = []
        # Recursive search for .pkl files
        for root, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if filename.endswith('.pkl'):
                    files.append(os.path.join(root, filename))
        
        if not files:
            print(f"No .pkl files found in {self.data_dir}")
            return []

        # Sort to ensure deterministic split
        files.sort()
        
        n_files = len(files)
        split_idx = int(0.8 * n_files)
        
        # Handle small datasets (e.g. < 5 files)
        if n_files > 0 and split_idx == 0:
            split_idx = n_files # Use all for train if very small
            
        if self.split == 'train':
            selected_files = files[:split_idx]
        elif self.split == 'val':
            selected_files = files[split_idx:]
        else:
            selected_files = files # Test uses all or specific subset
            
        print(f"Found {len(files)} total files. Selected {len(selected_files)} for {self.split} split.")
        return selected_files

    def _load_data(self):
        data_chunks = []
        for file_path in self.motion_files:
            # file_path is already absolute/relative from search
            try:
                with open(file_path, 'rb') as f:
                    motion_data = pickle.load(f)
                
                # Assuming motion_data is a dictionary or array with 'pose' or 'smpl_poses'
                if isinstance(motion_data, dict):
                    poses = motion_data.get('smpl_poses', motion_data.get('pose', None))
                else:
                    poses = motion_data # Assume array
                
                if poses is None:
                    continue

                # Slice into windows
                n_frames = poses.shape[0]
                if n_frames < self.window_size:
                    continue # Skip if too short
                    
                for i in range(0, n_frames - self.window_size + 1, self.stride):
                    window = poses[i : i + self.window_size]
                    data_chunks.append(window)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
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
