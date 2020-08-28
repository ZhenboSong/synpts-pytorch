import torch
import torch.nn as nn
import torch.optim as optim
from .networks import Discriminator, PtEncoder, ImgDecoder, RefineGenerator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss
import models.geometry as geo


class BaseModel2(nn.Module):
    def __init__(self, config):
        super(BaseModel2, self).__init__()

        self.config = config
        self.name = config.model_name

    def load(self):
        print('Loading %s generator Inpainting:' % self.name, self.config.refine_model)
        if torch.cuda.is_available():
            data = torch.load(self.config.refine_model)
        else:
            data = torch.load(self.config.refine_model, map_location=lambda storage, loc: storage)

        self.generator.load_state_dict(data['generator'])
        if self.config.mode == 'train':
            print('Loading %s discriminator:' % self.name, self.config.refine_model)
            if torch.cuda.is_available():
                data = torch.load(self.config.dis_model)
            else:
                data = torch.load(self.config.dis_model, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self, gen_path, dis_path):
        torch.save({
            'generator': self.generator.state_dict()
        }, gen_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, dis_path)


class BaseModel3(nn.Module):
    def __init__(self, config):
        super(BaseModel3, self).__init__()

        self.config = config
        self.name = config.model_name

    def load(self):
        print('Loading %s generator point cloud encoder:' % self.name, self.config.pte_model)

        if torch.cuda.is_available():
            data = torch.load(self.config.pte_model)
        else:
            data = torch.load(self.config.pte_model, map_location=lambda storage, loc: storage)

        self.pt_encoder.load_state_dict(data['pt_encoder'])

        print('Loading %s generator image decoder:' % self.name, self.config.imgd_model)

        if torch.cuda.is_available():
            data = torch.load(self.config.imgd_model)
        else:
            data = torch.load(self.config.imgd_model, map_location=lambda storage, loc: storage)

        self.img_decoder.load_state_dict(data['img_decoder'])

        if self.config.mode == 'train':
            print('Loading %s discriminator:' % self.name, self.config.dis_model)

            if torch.cuda.is_available():
                data = torch.load(self.config.dis_model)
            else:
                data = torch.load(self.config.dis_model, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self, pte_path, imgd_path, dis_path):
        torch.save({
            'pt_encoder': self.pt_encoder.state_dict()
        }, pte_path)

        torch.save({
            'img_decoder': self.img_decoder.state_dict()
        }, imgd_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, dis_path)


class Point2ImageModel(BaseModel3):
    def __init__(self, config):
        super(Point2ImageModel, self).__init__(config)

        self.im_size = config.image_size
        pt_encoder = PtEncoder(cfg=config)
        img_decoder = ImgDecoder()
        self.add_module('pt_encoder', pt_encoder)
        self.add_module('img_decoder', img_decoder)

        if config.mode == 'train':
            discriminator = Discriminator(in_channels=3, use_sigmoid=config.gan_loss != 'hinge')

            l1_loss = nn.L1Loss()
            perceptual_loss = PerceptualLoss()
            style_loss = StyleLoss()
            adversarial_loss = AdversarialLoss(type=config.gan_loss)

            self.add_module('discriminator', discriminator)

            self.add_module('l1_loss', l1_loss)
            self.add_module('perceptual_loss', perceptual_loss)
            self.add_module('style_loss', style_loss)
            self.add_module('adversarial_loss', adversarial_loss)

            self.pt_optimizer = optim.Adam(
                params=pt_encoder.parameters(),
                lr=float(config.learning_rate),
                betas=(config.beta1, config.beta2)
            )

            self.img_optimizer = optim.Adam(
                params=img_decoder.parameters(),
                lr=float(config.learning_rate),
                betas=(config.beta1, config.beta2)
            )

            self.dis_optimizer = optim.Adam(
                params=discriminator.parameters(),
                lr=float(config.learning_rate) * float(config.dis2gen_lr),
                betas=(config.beta1, config.beta2)
            )

            self.pt_scheduler = optim.lr_scheduler.StepLR(self.pt_optimizer,
                                                           step_size=config.lr_decay_epochs,
                                                           gamma=config.lr_decay_ratio)
            self.img_scheduler = optim.lr_scheduler.StepLR(self.img_optimizer,
                                                          step_size=config.lr_decay_epochs,
                                                          gamma=config.lr_decay_ratio)
            self.dis_scheduler = optim.lr_scheduler.StepLR(self.dis_optimizer,
                                                           step_size=config.lr_decay_epochs,
                                                           gamma=config.lr_decay_ratio)
            self.iteration = 0

    def lr_step(self):
        self.pt_scheduler.step()
        self.img_scheduler.step()
        self.dis_scheduler.step()

    def forward(self, ptc, T, K):
        # process outputs
        pt3 = geo.transform3Dto3D(ptc[:, :3, :], T)
        ptc_new = torch.cat((pt3, ptc[:, 3:, :]), dim=1)
        pt_feature, pt_new = self.pt_encoder(ptc_new.transpose(1, 2))

        # generate images
        img_tensor = list()
        for i in range(len(pt_new)):
            pt_trans = geo.transform3Dto3D(pt_new[i].transpose(1, 2), T)
            img_size = [int(self.im_size[0] / (1 << i)),
                        int(self.im_size[1] / (1 << i))]
            img_k = K / (1 << i)
            img_tensor.append(geo.projectPointFeature2Image(pt_trans, pt_feature[i], img_k, img_size))

        outputs = self.img_decoder(img_tensor)
        return outputs, img_tensor

    def process(self, ptc, T, K, real_img):
        self.iteration += 1

        # zero optimizers
        self.pt_optimizer.zero_grad()
        self.img_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs, img_tensor = self(ptc, T, K)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = real_img
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.adversarial_loss_w
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, real_img) * self.config.l1_loss_w
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, real_img)
        gen_content_loss = gen_content_loss * self.config.content_loss_w
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs, real_img)
        gen_style_loss = gen_style_loss * self.config.style_loss_w
        gen_loss += gen_style_loss

        # create logs
        logs = {
            "l_d2": dis_loss.item(),
            "l_g2": gen_gan_loss.item(),
            "l_l1": gen_l1_loss.item(),
            "l_per": gen_content_loss.item(),
            "l_sty": gen_style_loss.item(),
        }

        return outputs, gen_loss, dis_loss, logs, img_tensor

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.pt_optimizer.step()
        self.img_optimizer.step()


class InpaintingModel(BaseModel2):
    def __init__(self, config):
        super(InpaintingModel, self).__init__(config)

        generator = RefineGenerator(7, 3)
        self.add_module('generator', generator)

        if self.config.mode == 'train':
            discriminator = Discriminator(in_channels=3, use_sigmoid=config.gan_loss != 'hinge')

            l1_loss = nn.L1Loss()
            perceptual_loss = PerceptualLoss()
            style_loss = StyleLoss()
            adversarial_loss = AdversarialLoss(type=config.gan_loss)

            self.add_module('discriminator', discriminator)

            self.add_module('l1_loss', l1_loss)
            self.add_module('perceptual_loss', perceptual_loss)
            self.add_module('style_loss', style_loss)
            self.add_module('adversarial_loss', adversarial_loss)

            self.gen_optimizer = optim.Adam(
                params=generator.parameters(),
                lr=float(config.learning_rate),
                betas=(config.beta1, config.beta2)
            )

            self.dis_optimizer = optim.Adam(
                params=discriminator.parameters(),
                lr=float(config.learning_rate) * float(config.dis2gen_lr),
                betas=(config.beta1, config.beta2)
            )

            self.gen_scheduler = optim.lr_scheduler.StepLR(self.gen_optimizer,
                                                           step_size=config.lr_decay_epochs,
                                                           gamma=config.lr_decay_ratio)
            self.dis_scheduler = optim.lr_scheduler.StepLR(self.dis_optimizer,
                                                           step_size=config.lr_decay_epochs,
                                                           gamma=config.lr_decay_ratio)
            self.iteration = 0

    def lr_step(self):
        self.gen_scheduler.step()
        self.dis_scheduler.step()

    def process(self, input_img, real_img):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(input_img)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = real_img
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.adversarial_loss_w
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, real_img) * self.config.l1_loss_w
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, real_img)
        gen_content_loss = gen_content_loss * self.config.content_loss_w
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs, real_img)
        gen_style_loss = gen_style_loss * self.config.style_loss_w
        gen_loss += gen_style_loss

        # create logs
        logs = {
            "l_d2": dis_loss.item(),
            "l_g2": gen_gan_loss.item(),
            "l_l1": gen_l1_loss.item(),
            "l_per": gen_content_loss.item(),
            "l_sty": gen_style_loss.item(),
        }

        return outputs, gen_loss, dis_loss, logs

    def forward(self, image):
        outputs = self.generator(image)
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


class Point2ImageRefine(nn.Module):
    def __init__(self, config):
        super(Point2ImageRefine, self).__init__()
        self.cfg = config
        self.p2i_model = Point2ImageModel(config)
        self.refine_mode = InpaintingModel(config)

    def load(self):
        self.p2i_model.load()
        self.refine_mode.load()

    def save(self, pte_path, imgd_path, refine_path, dis_path):
        if not self.cfg.only_train_refine:
            self.p2i_model.save(pte_path, imgd_path, dis_path)
        self.refine_mode.save(refine_path, dis_path)

    def process(self,  ptc, T, K, real_img):
        coarse_img, gen_coarse_loss, dis_coarse_loss, logs_coarse, img_tensor = self.p2i_model.process(ptc, T, K, real_img)
        pt3 = geo.transform3Dto3D(ptc[:, :3, :], T)
        z = pt3[:, 2:3, :]
        addition_z = geo.projectPointFeature2Image(pt3, z / 5.0, K, self.cfg.image_size)
        addition_rgb = geo.projectPointFeature2Image(pt3, ptc[:, 3:, :], K, self.cfg.image_size)
        for_refine_image = torch.cat((coarse_img.detach(), addition_rgb, addition_z), dim=1)
        outputs, gen_loss, dis_loss, logs = self.refine_mode.process(for_refine_image, real_img)
        return [coarse_img, outputs], [gen_coarse_loss, gen_loss], [dis_coarse_loss, dis_loss], \
               [logs_coarse, logs], img_tensor

    def backward(self, gen_loss=None, dis_loss=None):
        if not self.cfg.only_train_refine:
            self.p2i_model.backward(gen_loss[0], dis_loss[0])
        self.refine_mode.backward(gen_loss[1], dis_loss[1])

    def forward(self, ptc, T, K):
        coarse_image, image_tensor = self.p2i_model(ptc, T, K)
        pt3 = geo.transform3Dto3D(ptc[:, :3, :], T)
        z = pt3[:, 2:3, :]
        addition_z = geo.projectPointFeature2Image(pt3, z / 5.0, K, self.cfg.image_size)
        addition_rgb = geo.projectPointFeature2Image(pt3, ptc[:, 3:, :], K, self.cfg.image_size)
        for_refine_image = torch.cat((coarse_image.detach(), addition_rgb, addition_z), dim=1)
        output = self.refine_mode(for_refine_image)
        return [coarse_image, output], image_tensor

    def lr_step(self):
        if not self.cfg.only_train_refine:
            self.p2i_model.lr_step()
        self.refine_mode.lr_step()

