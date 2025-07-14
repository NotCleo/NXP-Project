    def preprocess_image(self, image):
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

    def xywh2xyxy(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def box_iou(self, box1, box2):
        def box_area(box):
            return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        area1 = box_area(box1)
        area2 = box_area(box2)
        inter = (
            torch.min(box1[:, None, 2:], box2[:, 2:]) -
            torch.max(box1[:, None, :2], box2[:, :2])
        ).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        bs = prediction.shape[0]
        nc = prediction.shape[2] - 5
        xc = prediction[..., 4] > conf_thres
        max_wh = 7680
        max_nms = 30000
        output = []
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                output.append(torch.zeros((0, 6)))
                continue
            x[:, 5:] *= x[:, 4:5]
            box = self.xywh2xyxy(x[:, :4])
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
            c = x[:, 5:6] * max_wh
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = []
            while boxes.size(0):
                max_idx = scores.argmax()
                i.append(max_idx.item())
                if len(i) >= max_det or boxes.size(0) == 1:
                    break
                iou = self.box_iou(boxes[max_idx].unsqueeze(0), boxes)[0]
                keep = iou <= iou_thres
                boxes = boxes[keep]
                scores = scores[keep]
                x = x[keep]
            if i:
                output.append(x[torch.tensor(i)])
            else:
                output.append(torch.zeros((0, 6)))
        return output
