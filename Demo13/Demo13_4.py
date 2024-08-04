import torch
from d2l import torch as d2l

torch.set_printoptions(2) # 精简输出精度

# 生成多个锚框
def multibox_prior(data, sizes, ratios):
    """ sizes:缩放比，ratios:宽高比
    生成sizes + ratios 个锚框"""
    # 得到高宽，因为可能是批量输出，故选最后
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    # 每个像素的锚框数量
    boxes_per_pixel = num_sizes + num_ratios - 1
    size_tensor = torch.tensor(sizes, device=device)
    ratios_tensor = torch.tensor(ratios, device=device)

    # 为了将锚框的中心点移动到像素的中心，需要设置偏移量
    # 因为1像素的高宽为1，故偏移量中心为0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    # 生成所有锚框的中心点并进行缩放
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    # 生成二维坐标:
    # y[[a a a ...] [b b b ...]...]
    # x[[a b ...] [a b ..]]
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    print(shift_y)
    print(shift_x)
    # 改成一串，没有行列
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    print(shift_x)
    print(shift_x)
    # 生成boxes_per_pixel个高和宽
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratios_tensor[0]),
                   sizes[0] * torch.sqrt(ratios_tensor[1:]))) \
                   * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratios_tensor[0]),
                   sizes[0] / torch.sqrt(ratios_tensor[1:])))
    # 除以2来获得半高和半宽
    # 这里用-w, -h, w, h是来获取每个锚框相对的位置（注意后面/2），然后转置并重复这么多次获得所有的锚框，这次重复这么多是为了后面直接生成坐标(xmin,xmax,ymin,ymax)
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2

    # 每个中心都有boxes_per_pixel个锚框
    # 所以生成含有锚框中心点的网格重复了boxes_per_pixel次
    # repeat_interleave逐元素重复
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    # 这里将框的中心点和锚框的尺寸加起来得到点
    out_put = out_grid + anchor_manipulations
    return out_put.unsqueeze(0)

img = d2l.plt.imread('catdog.jpg')
h, w = img.shape[:2]
print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)
print(Y[0][0])

# 画框函数
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes.
    Defined in :numref:`sec_anchor`"""
    # 这个函数用来将输入对象转换成列表，如果输入对象是 None，则返回默认值，如果输入对象不是列表或元组，则将其转换成列表。
    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            # 如果提供了标签，并且标签数量大于当前索引：则在矩形的左上角添加标签。文本颜色取决于矩形的颜色，如果矩形的颜色是白色，则文本颜色为黑色，否则为白色。
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


boxes = Y.reshape(h, w, 5, 4)
d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
d2l.plt.show()

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """这里与理论有点不同，是先进行第二步，再进行第一步，这个最后可以替代掉前面的最大值"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界j的Iou
    jaccard = d2l.box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)

    # 根据阈值，决定是否分配真是边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    # 返回非零坐标,筛选 IoU 大于等于阈值的锚框索引及其对应的真实边界框索引
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    # 每个锚框对应分配的真实边界值
    anchors_bbox_map[anc_i] = box_j
    # 确保每个真实边界框至少分配到一个锚框
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    # 开始循环 num_gt_boxes 次（确保每个真实边界框至少分配一个锚框）：
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        # 确定分配的真实边界框，求余，.long()代表分配整数
        box_idx = (max_idx % num_gt_boxes).long()
        # 确定分配的锚框
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard

    return anchors_bbox_map

#@save
def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    # anchors：一组锚框，形状为 (1, 4, 4)（简化为 4 个锚框）
    # labels：真实标签，形状为 (1, 2, 5)（一个样本，两个真实边界框，每个边界框包含类别标签和 4 个坐标）
    # anchors = np.array([
    #     [[0.1, 0.1, 0.2, 0.2],
    #      [0.2, 0.2, 0.3, 0.3],
    #      [0.3, 0.3, 0.4, 0.4],
    #      [0.4, 0.4, 0.5, 0.5]]
    # ])
    #
    # labels = np.array([
    #     [[1, 0.15, 0.15, 0.25, 0.25],
    #      [2, 0.35, 0.35, 0.45, 0.45]]
    # ])
    # 这里用squeeze(0)代表降维成1维
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        # 在最后一个维度插入一个大小为 1 的新维度。例如，如果原始张量的形状是 (num_anchors,)，经过 .unsqueeze(-1) 处理后，形状变为 (num_anchors, 1)
        # 如果 x 的形状为 (m, n)，那么 x.repeat(a, b) 的形状将变为 (a * m, b * n)。
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        # 加1是因为有背景0类别
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = d2l.offset_boxes(anchors, assigned_bb) * bbox_mask
        # 变成一维并追加
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    # 第一次循环示例：
    #label = labels[0, :, :]  # [[1, 0.15, 0.15, 0.25, 0.25], [2, 0.35, 0.35, 0.45, 0.45]]
    # # 分配锚框
    # anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
    # # 假设分配结果是 [0, 1, 1, -1]，表示第一个锚框对应第一个真实边界框，
    # # 第二、第三个锚框对应第二个真实边界框，第四个锚框没有对应边界框。
    #
    # bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
    # # bbox_mask = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
    #
    # class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
    # assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
    # # class_labels = [0, 0, 0, 0]
    # # assigned_bb = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    #
    # indices_true = torch.nonzero(anchors_bbox_map >= 0).squeeze()
    # # indices_true = [0, 1, 2]
    #
    # bb_idx = anchors_bbox_map[indices_true]
    # # bb_idx = [0, 1, 1]
    #
    # class_labels[indices_true] = label[bb_idx, 0].long() + 1
    # # class_labels = [2, 3, 3, 0]
    #
    # assigned_bb[indices_true] = label[bb_idx, 1:]
    # # assigned_bb = [[0.15, 0.15, 0.25, 0.25], [0.35, 0.35, 0.45, 0.45], [0.35, 0.35, 0.45, 0.45], [0, 0, 0, 0]]
    #
    # # 偏移量转换
    # offset = offset_boxes(anchors, assigned_bb) * bbox_mask
    # # 计算出每个锚框的偏移量，例如：
    # # offset = [[dx1, dy1, dx2, dy2], ...]
    # # bbox_offset = 堆叠后的偏移量张量
    # # bbox_mask = 堆叠后的掩码张量
    # # class_labels = 堆叠后的类别标签张量
    return (bbox_offset, bbox_mask, class_labels)


