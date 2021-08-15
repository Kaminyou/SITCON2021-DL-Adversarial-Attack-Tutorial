import torch

class Attacker:
    def __init__(self, model, data, target, criterion, device="cuda"):
        self.data = data
        self.target = target   
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.criterion = criterion

    # FGSM
    def fgsm_attack(self, data, epsilon, data_grad):

        sign_data_grad = data_grad.sign()
        perturbed_data = data - epsilon * sign_data_grad 
        return perturbed_data
    
    # PGD
    def pgd_attack(self, data, ori_data, eps, alpha, data_grad) :
        
        adv_data = data - alpha * data_grad.sign() # + -> - !!!
        eta = torch.clamp(adv_data - ori_data.data, min=-eps, max=eps)
        data = ori_data + eta

        return data
    
    def attack(self, epsilon, alpha, attack_type = "FGSM", PGD_round=40):
        data, target = self.data.to(self.device), self.target.to(self.device)
        data_raw = data.clone().detach()

        ############ ATTACK GENERATION ##############
        if attack_type == "FGSM":
            data.requires_grad = True
            pred = self.model(data)
            loss = self.criterion(pred, target)
            
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

        elif attack_type == "PGD":
            for i in range(PGD_round):
                #print(f"PGD processing ...  {i+1} / {PGD_round}       ", end="\r")
                data.requires_grad = True
                
                pred = self.model(data)
                loss = self.criterion(pred, target)
                
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data

                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()
            perturbed_data = data

        perturbed_data = perturbed_data.detach()
        return perturbed_data