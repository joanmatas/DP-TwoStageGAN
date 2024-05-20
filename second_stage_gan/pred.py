from connect import get_direction_list, get_random_enter_exit_point, to_seq
import os
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
np.set_printoptions(threshold=sys.maxsize)

# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)

MIN_LON=-8.687466
MIN_LAT=41.123232
MAX_LON=-8.553186
MAX_LAT=41.237424

def transform(img, enter, esc):
    img = cv.resize(img, (512, 512), cv.INTER_NEAREST)
    img = img.astype(np.float32)[:, :, ::-1]
    img = img / 255.0
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    img = img.transpose((2, 0, 1))
    img = torch.tensor(img, dtype=torch.float).reshape(1, 3, 512, 512)
    enter_point = torch.tensor(enter, dtype=torch.float)  # dim = 2
    esc_point = torch.tensor(esc, dtype=torch.float)  # dim = 2
    return img, enter_point, esc_point


def predict(decoder, encoder, img, enter, esc):
    img, enter, esc = transform(img, enter, esc)
    max_len = 8
    batch_size = 1
    decoder.eval()
    encoder.eval()
    length = torch.full((1, 1), max_len, dtype=torch.long)
    # Move to device, if available
    img = img.to(device)  # (b,c,w,h)
    enter = enter.to(device)  # (b,2)
    esc = esc.to(device)  # (b,2)
    encoder_out = encoder(img)
    encoder_dim = encoder_out.size(-1)
    # Flatten image
    # (b, num_pixels, encoder_dim)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)  # for attention. not useful at the moment

    # Initialize LSTM state
    h, c, h_inv, c_inv = decoder.init_hidden_state(encoder_out, enter, esc)

    # Create tensors to hold two coordination predictions
    predictions_ord = torch.zeros((batch_size,max_len,2)).to(device)  # (b,max_len,2)
    predictions_inv = torch.zeros((batch_size,max_len,2)).to(device)  # (b,max_len,2)

    predictions_ord[:,0,:] = enter
    predictions_inv[:,max_len-1,:] = esc

    for t in range(max_len):
        h_a = torch.cat([h,h_inv],dim=1)
        h_b = torch.cat([h_inv,h],dim=1)
        c_a = torch.cat([c,c_inv],dim=1)
        c_b = torch.cat([c_inv,c],dim=1)

        #attention_weighted_encoding, alpha = decoder.attention(encoder_out,h_a)
        #attention_weighted_encoding_inv, alpha_inv = decoder.attention(encoder_out,h_b)
        #gate = decoder.sigmoid(decoder.f_beta(h_a))
        #gate_inv = decoder.sigmoid(decoder.f_beta(h_b))
        #attention_weighted_encoding = gate * attention_weighted_encoding
        #attention_weighted_encoding_inv = gate_inv * attention_weighted_encoding_inv

        # weight is attention (differ from var weights below)
        weight = F.softmax(decoder.attention(h_a), dim=1) # weight for each input pixels
        weight_inv = F.softmax(decoder.attention(h_b), dim=1) # (batch_size_t,n_pixels)

        h, c = decoder.decoder(
            torch.cat([decoder.position_embedding(predictions_ord[:,t,:]),encoder_out[:,t,:] * weight],dim=1),
            (h_a, c_a))  # (batch_size_t, decoder_dim)

        h_inv, c_inv = decoder.decoder_inv(
            torch.cat([decoder.position_embedding(predictions_inv[:,max_len-1-t,:]),encoder_out[:,t,:] * weight_inv],dim=1),
            (h_b, c_b))

        h = decoder.trans_h(h)
        c = decoder.trans_c(c)
        h_inv = decoder.trans_h(h_inv)
        c_inv = decoder.trans_c(c_inv)

        preds = decoder.fc(decoder.dropout(h))  # (batch_size_t, 2)
        preds_inv = decoder.fc(decoder.dropout(h_inv))
        if t < max_len - 1:
            predictions_ord[:, t + 1, :] = preds # (b,max_len,2)
            predictions_inv[:, max_len-2-t,:] = preds_inv

    ## weight scheme 2
    weights = np.array([_ for _ in range(max_len)])
    weights = np.exp(-weights)
    weights_inv = weights[::-1]
    weights = np.vstack([weights,weights_inv])
    weights /= weights.sum(axis=0)
    weights_inv = weights[1,:]
    weights = weights[0,:]

    weights = torch.tensor(weights,dtype=torch.float).to(device).unsqueeze(0).unsqueeze(2)
    weights_inv = torch.tensor(weights_inv, dtype=torch.float).to(device).unsqueeze(0).unsqueeze(2)
    predictions = (predictions_ord * weights + predictions_inv * weights_inv)
    pred_max = torch.max(predictions)
    pred_min = torch.min(predictions)

    return torch.div(predictions, max(pred_max, - pred_min))


def to_coordinate(name, pred):
    [i, j] = name.split('_')
    i, j = int(i), int(j)
    i = 31 - i
    pred = np.add(pred / 2, np.array([j, i]))
    # pred=np.array([[j, i]])
    pred = pred*np.array([(MAX_LON-MIN_LON)/32,(MAX_LAT-MIN_LAT)/32]) + np.array([MIN_LON, MIN_LAT])
    return pred


def main():
    step_1_output = '../data/generated_coarse'
    #    map_name = 'map.png'
    map_dir = '../data/pics'
    checkpoint = '../output/2ndGAN/checkpoint_epoch_42_loss:0.35.pth'
    to_seq_tries = 5
    #map_whole = cv.imread(map_name)
    #map_grids = grid(map_whole)  # map_grids[point]; point = '1_2'
    print('loading model ...')
    checkpoint = torch.load(checkpoint)
    decoder = checkpoint['decoder']
    encoder = checkpoint['encoder']
    count = 0
    n = 70
    print(n)
    for grid_data_num in tqdm(range(3100 + 300 * n, 3100 + 300 * (n + 1))):
        grid_data_name = str(grid_data_num) + '.npy'
        traj = []
        flag = True
        try:
            generated_traj_mat = np.load(os.path.join(
                step_1_output, grid_data_name), allow_pickle=True)
        except OSError:
            continue

        if not np.any(generated_traj_mat[0]):
            count += 1
            # print('matrix drop out: %d' % (count))
            continue

        # choose the longest obtained sequence in to_seq_tries tries
        grid_data = []
        for i in range(to_seq_tries):
            new_grid_data = to_seq(generated_traj_mat)    # [[x0, y0, t0], [x1, y1, t1], ... ]
            if len(new_grid_data) > len(grid_data):
                grid_data = new_grid_data

        if len(grid_data) <= 2:
            count += 1
            # print('sequence drop out: %d' % (count))
            # print(grid_data)
            continue

        enter_list, exit_list = get_direction_list(grid_data)
        enter_point, exit_point = None, None

        for grid_point, enter, exit in zip(grid_data, enter_list, exit_list):
            grid_name = '%d_%d' % (grid_point[0], grid_point[1])
            grid_pic_name = '%d_%d' % (grid_point[1], 31-grid_point[0])
            #map_grid = map_grids[grid_name]
            grid_pic = grid_pic_name + '.png'
            if grid_pic in os.listdir(map_dir):
                map_grid = cv.imread(map_dir+grid_pic)
                # print('Load map %s'%(grid_pic))
            else:
                flag = False
                break
            enter_point, exit_point = get_random_enter_exit_point(
                map_grid, enter, exit, 0.5, exit_point)
            pred = predict(decoder, encoder, map_grid, enter_point, exit_point)
            del map_grid
            pred = pred.detach().cpu().numpy().reshape((8, 2))
            pred = to_coordinate(grid_name, pred)
            traj.append(pred)
            del pred
        if flag == False:
            continue
        traj = np.concatenate(traj, axis=0)
        text_file_name = '../output/final_generated_data/text_files/' + grid_data_name[:-4] + '.txt'
        with open(text_file_name, 'w') as file:
            for coord in traj:
                file.write(str(coord[1]) + ',' + str(coord[0]) + ',\n')

        # print('Save trajectory: %s'%(grid_data_name))
        np.save(os.path.join('../output/final_generated_data/numpys/', grid_data_name), traj)
        del traj


if __name__ == '__main__':
    main()