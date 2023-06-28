
def process_resnet_output(annot_length, batch_images, batch_annotations, output, ):
    padded_outputs = []
            tensor1 = torch.zeros((1,annot_length))
            tensor2 = torch.zeros((1,annot_length))
            tensor3 = torch.zeros((1,annot_length,4))
            for i in range(len(batch_images)):
                #print('-------------------------------------', i)
                output_sized = output[i] #.unsqueeze(0)
                #print('output_sized',output_sized)

                padded_boxes = torch.nn.functional.pad(output_sized['boxes'], pad=(0, 0, 0, annot_length - len(output_sized['boxes'])), mode='constant', value=0)
                padded_scores = torch.nn.functional.pad(output_sized['scores'], pad=(0, annot_length - len(output_sized['scores'])), mode='constant', value=0)
                padded_labels = torch.nn.functional.pad(output_sized['labels'], pad=(0, annot_length - len(output_sized['labels'])), mode='constant', value=0)
                padded_output = {'boxes': padded_boxes, 'scores': padded_scores, 'labels': padded_labels}

                #print('padded_output', padded_output)
                padded_outputs.append(padded_output)

                annot_labels = batch_annotations['labels'][i,:]
                #annot_boxes = batch_annotations['boxes']
                #print('annot_boxes', annot_boxes.shape)
                #annot_box_i = annot_boxes[i,:,:]
                #print('annot_box_i', annot_box_i)
                #print(padded_labels)

                batch_annotations_y = torch.eq(annot_labels, padded_labels).int().unsqueeze(0)
                #print(batch_annotations_y.shape)
                #print(tensor1.shape)
                #print(tensor1)
                tensor1 = torch.cat((tensor1, batch_annotations_y), dim=0)
                #print('tensor1', tensor1)

                
                tensor2 = torch.cat((tensor2, padded_scores.unsqueeze(0)), dim=0)
                #print('tensor2', tensor2)

                padded_boxes = padded_boxes.unsqueeze(0)
                #print('padded_boxes', padded_boxes.shape)
                
                annot_boxes = batch_annotations['boxes'][i,:,:]
                #print(annot_boxes.shape)
                
                # L_cls = nn.SmoothL1Loss()
                # loss = L_cls(padded_boxes, annot_boxes)
                # print('loss', loss)
                # print('L_cls',L_cls)
                #print(tensor3.shape)
                tensor3 = torch.cat((tensor3, padded_boxes), dim=0)

            #print(tensor3.shape)
            #rint
            #print('tensor3', tensor3)
            output_boxes = tensor3[1:,:,:]
            #print('output_boxes', output_boxes.shape)
            annot_boxes = batch_annotations['boxes']
            #print('annot_boxes', annot_boxes.shape)
        
            batch_scores = tensor2[1:]
            #print(batch_scores)
            batch_annotations_y = tensor1[1:]
            #print(batch_annotations_y)
            #print('padded_outputs', padded_outputs)
            

            #print(batch_annotations['labels'])
            #print(padded_scores.unsqueeze(0).shape)
            #print(padded_labels.unsqueeze(0))
            # Two losfunctions are used
            # Regression loss = alpha balanced focal loss
            # classification loss = smooth-L1 loss
            #print(batch_scores.shape)
            #print(batch_annotations_y.shape)

            return output_boxes, annot_boxes, batch_scores, batch_annotations_y