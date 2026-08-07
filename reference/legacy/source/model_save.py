import os
import torch

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
        self.file_name = file_name
        self.log_file_path = os.path.join(opt.save_path, 'val', 'val_log_perf_' + 'model_pl_l1' + '.txt')
        self.model_dir_path = os.path.join(opt.save_model, 'saved_model_model_pl_l1')
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
            lines = log_file.readlines()

        # Check if there are at least two lines (header + one entry)
        if len(lines) >= 2:
            best_psnr_model = None #None  # (epoch, PSNR) pair for the top PSNR model
            best_ssim_model = None  # (epoch, SSIM) pair for the top SSIM model
            best_psnr_value = 0
            best_ssim_value = 0

            for line in lines[1:]:  # Skip the header line
                metrics = self.parse_metrics(line)
                if metrics:
                    epoch, psnr, ssim = metrics[0], metrics[1], metrics[2]
                    #print("metrics:", epoch, psnr, ssim)

                    # Check if PSNR is higher than the current best
                    if psnr > best_psnr_value:
                        best_psnr_model = (epoch, psnr)
                        best_psnr_value = psnr


                    # Check if SSIM is higher than the current best
                    if ssim > best_ssim_value:
                        best_ssim_model = (epoch, ssim)
                        best_ssim_value = ssim  

            
            for root, _, files in os.walk(self.model_dir_path):
                for file in files:
                    dir_epoch = float(file.split("_")[2].split("#")[0])
                    for metric in (psnr, ssim):
                        if ((dir_epoch, metric)) not in [best_psnr_model, best_ssim_model]:
                            dir_to_delete = os.path.join(root, file)
                            
                            if os.path.exists(dir_to_delete):
                                os.remove(dir_to_delete)
            
            
            
            # Save the best models based on PSNR and SSIM
            if best_psnr_model:
                torch.save(self.generator.state_dict(), 
                            os.path.join(self.model_dir_path, f'generator_PSNR_{best_psnr_model[0]}#.pkl'))
            if best_ssim_model:
                torch.save(self.generator.state_dict(), 
                            os.path.join(self.model_dir_path, f'generator_SSIM_{best_ssim_model[0]}#.pkl'))

            # Delete all model directories except for the top two
            
            
            # for root, _, files in os.walk(self.model_dir_path):
            #     for file in files:
            #         dir_epoch = float(file.split("_")[2].split("#")[0])
            #         for metric in (psnr, ssim):
            #             if ((dir_epoch, metric)) not in [best_psnr_model, best_ssim_model]:
            #                 dir_to_delete = os.path.join(root, file)
            #                 if os.path.exists(dir_to_delete):
            #                     os.system("rm -r " + dir_to_delete)
            #                 # print(f"Deleted model directory: {dir_to_delete}")

        else:
            print("Log file does not contain enough data for comparison")
            
# if __name__ == '__main__':
#     from param import Opts
#     opt = Opts()
#     model_saver = ModelSaver(opt=opt) #generator=self.generator, 
#     model_saver.save_best_models() 