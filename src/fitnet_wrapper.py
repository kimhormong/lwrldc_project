import torch
import torch.nn as nn

class FitNetWrapper(nn.Module):
    def __init__(self, teacher, student, hint_layer_name, guided_layer_name, student_channels, teacher_channels):
        super(FitNetWrapper, self).__init__()
        self.teacher = teacher
        self.student = student
        self.regressor = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1),
            nn.BatchNorm2d(teacher_channels),
            nn.ReLU(inplace=True)
        )
        self.hint_output = None
        self.guided_output = None
        self._register_hooks(hint_layer_name, guided_layer_name)

    def _register_hooks(self, hint_name, guided_name):
        def get_activation(is_teacher):
            def hook(model, input, output):
                if is_teacher: self.hint_output = output
                else: self.guided_output = output
            return hook
        dict(self.teacher.named_modules())[hint_name].register_forward_hook(get_activation(True))
        dict(self.student.named_modules())[guided_name].register_forward_hook(get_activation(False))

    def forward(self, x):
        s_logits = self.student(x)
        with torch.no_grad():
            t_logits = self.teacher(x)
    
        # Get raw guided features
        guided_features = self.guided_output 
    
        # 1. Apply convolutional regressor (fixes channels) 
        # apply the regressor to make spatial features from student match teacher's hint number of channels
        regressed_student = self.regressor(guided_features)
    
    
        # 2. Fix Spatial Mismatch (Height/Width)
        # Teacher Spatial Size: 14x14x256
        # Student Spatial Size: 14x14x80
        # Width and height from both teacher and student are 14x14, so we can directly compare them without resizing.
        # TODO: If there is a mismatch, section 2.2 HINT-BASED TRAINING of the paper (FITNETS: HINT-BASED TRAINING FOR DEEP NETWORKS) provides guidance on how to handle it

        
        return s_logits, t_logits, regressed_student, self.hint_output