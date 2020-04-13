import os
var_string = '--batchSize 16 --originalSize 800 --originalSize 600 --epoch 20 --mode test --learningRate 0.0003 --numclass 2'
resize = [256, 320]
cof = [16, 32, 64, 128]
down_up_conv = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
# down_up_conv = [(5, 5), (6, 6)]
for r in resize:
    for c in cof:
        for du in down_up_conv:
            var_string += ' --resize ' + str(r) + ' --resize ' + str(r) + ' --cof  ' + str(c) + ' --downLayerNumber ' + str(du[0])+ ' --upLayerNumber ' + str(du[1])
            script = ('python /content/cloned-repo/seg/train.py ' + var_string)
            result = os.system(script)
            if result ==0 :
              print('参数组合%s训练完成' % (var_string))
              var_string = '--batchSize 16 --originalSize 800 --originalSize 600 --epoch 20 --mode test --learningRate 0.0003 --numclass 2'
            else:
              print('执行参数组合%s时出错，报错信息%s跳过..' % (var_string, result))
