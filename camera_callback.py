def camera_image_callback(self, message):
    """Callback function to handle incoming camera images and run YOLO inference."""
    np_arr = np.frombuffer(message.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    input_img = self.preprocess_image(image)

    
	# This below line does the Detection first
	qr_detector = cv2.QRCodeDetector()

	# This below line does the decoding after detection (qr_detector)
	data, points, _ = qr_detector.detectAndDecode(image)

	if data:
		self.qr_code_str = data
		self.get_logger().info(f"QR Code Data: {data}")

		if points is not None:
			points = points.astype(int)
			for i in range(len(points[0])):
				pt1 = tuple(points[0][i])
				pt2 = tuple(points[0][(i + 1) % len(points[0])])
				cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=3)

    if self.int8:
        scale, zero_point = self.input_details[0]["quantization"]
        input_img = (input_img / scale + zero_point).astype(np.uint8)

    self.interpreter.set_tensor(self.input_details[0]["index"], input_img)
    self.interpreter.invoke()

    output_data = []
    for out in self.output_details:
        out_tensor = self.interpreter.get_tensor(out["index"])
        if self.int8:
            scale, zero_point = out["quantization"]
            out_tensor = (out_tensor.astype(np.float32) - zero_point) * scale
        output_data.append(out_tensor)

    detected_objects = {}
    for pred in output_data:
        pred[0][..., :4] *= [self.input_size, self.input_size, self.input_size, self.input_size]
        torch_pred = torch.tensor(pred)
        detections = self.non_max_suppression(torch_pred, 0.25, 0.45, max_det=1000)
        for det in detections:
            if det is None or not len(det):
                continue
            for *xyxy, conf, cls in det:
                label = self.label_names[int(cls)] if int(cls) < len(self.label_names) else str(int(cls))
                detected_objects[label] = detected_objects.get(label, 0) + 1

    if detected_objects:
        shelf_data_message = WarehouseShelf()
        shelf_data_message.object_name = list(detected_objects.keys())
        shelf_data_message.object_count = list(detected_objects.values())
        shelf_data_message.qr_decoded = self.qr_code_str

        self.publisher_shelf_data.publish(shelf_data_message)

        # OPTIONAL: publish debug image
        self.publish_debug_image(self.publisher_qr_decode, image)
