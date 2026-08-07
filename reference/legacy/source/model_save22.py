import os
import torch
import shutil

# import sys

# # Add the folder containing the module to sys.path
# sys.path.append('/gpfs/fs0/scratch/u/uanazodo/uanazodo/Ray/SMART_V1.0')

# # Import the module
# import param
# from param import Opts
# opt = Opts()

class ModelSaver:
    def __init__(self, generator, opt, file_name): #generator
        self.opt = opt
        self.file_name =file_name
        self.log_file_path = os.path.join(opt.save_path, 'val', 'val_log_perf_' + self.file_name + '.txt').replace("main", "model")
        self.model_dir_path = os.path.join(opt.save_path, 'saved_model_' + self.file_name)
        self.generator = generator
        

    # Function to parse the performance metrics from a line
    @staticmethod
    def parse_metrics(line):
        parts = line.split("\t")
        parts = [item for item in parts if item != '\n']
        if len(parts) > 1:
            epoch, psnr, ssim, nrmse, mse = map(float, parts)
            return epoch, psnr, ssim, nrmse, mse
        else:
            return None

    def save_best_models(self):
        # Create folder to save models 
        os.makedirs(self.model_dir_path, exist_ok=True)
        # Read the log file
        with open(self.log_file_path, "r") as log_file:
            next(log_file)
            lines = log_file.readlines()

        # Check if there are at least two lines (header + one entry)
        if len(lines) >= 2:
            print(lines)
            latest_metrics = self.parse_metrics(lines[-1])
            second_latest_metrics = self.parse_metrics(lines[-2])

            if latest_metrics and second_latest_metrics:
                latest_epoch, latest_psnr, latest_ssim, _, _ = latest_metrics
                second_latest_epoch, second_latest_psnr, second_latest_ssim, _, _ = second_latest_metrics

            if latest_psnr > second_latest_psnr:
                # Delete the last saved model if it exists
                last_saved_model_path = os.path.join(self.model_dir_path, f'generator_PSNR_{second_latest_epoch}.pkl')
                if os.path.exists(last_saved_model_path):
                    os.remove(last_saved_model_path)
                    
                # Save the new best model
                torch.save(self.generator.state_dict(), os.path.join(self.model_dir_path, f'generator_PSNR_{latest_epoch}.pkl'))
                    
            if latest_ssim > second_latest_ssim:
                # Delete the last saved model if it exists
                last_saved_model_path = os.path.join(self.model_dir_path, f'generator_SSIM_{second_latest_epoch}.pkl')
                if os.path.exists(last_saved_model_path):
                    os.remove(last_saved_model_path)

                # Save the new best model
                torch.save(self.generator.state_dict(), os.path.join(self.model_dir_path, f'generator_SSIM_{latest_epoch}.pkl'))

            
# if __name__ == '__main__':
#     from param import Opts
#     opt = Opts()
#     model_saver = ModelSaver(opt=opt) #generator=self.generator, 
#     model_saver.save_best_models() 