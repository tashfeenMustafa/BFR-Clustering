import random
import math
import json
import sys
import time
import numpy as np


def load(file_name):
    # data(list of list): [[index, dimensions], [.., ..], ...]
    data = np.loadtxt(file_name, dtype="object", delimiter=",")
    for i in range(len(data)):
        for j in range(len(data[i])):
            if j == 0:
                data[i][j] = int(data[i][j])
            else:
                if data[i][j] == None:
                    data[i][j] = float(1)
                else:
                    data[i][j] = float(data[i][j])
                    if math.isnan(data[i][j]):
                        data[i][j] = float(1)
    return data


def initialize_centroids_simple(data, dimension, k):
    # centroids: [(centroid0 fearures); (centroid0 features); ... ..]
    centroids = np.empty((k, dimension))
    # TO DO
    # Write your code to return initialized centroids by randomly assiging them to K points
    # Select 'k' points from data and put it centroids
    for i in range(k):
        index = random.randint(0, dimension)
        print(data[index])
        centroids[i] = data[index][1:]
    return centroids


def initialize_centroids(data, dimension, k):
    centroids = np.empty((k, dimension))

    max_feature_vals = np.empty((0, dimension))
    min_feature_vals = np.empty((0, dimension))

    # TO DO
    # Calculate max feature and min feture value for each dimension    
    for i in range(dimension):
        min_feature_vals[i] = min(data[:, i])
        max_feature_vals[i] = max(data[:, i])

    # diff: max - min for each dimension
    diff = max_feature_vals - min_feature_vals

    # for each centroid, in each dimension assign centroids[j][i] = min_feature_val + diff * random.uniform(1e-5, 1)   
    for i in range(k):
        centroids[i] = min_feature_vals + diff * random.uniform(1e-5, 1)

    return centroids


def get_euclidean_distance(p1, p2):
    # Write your code
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    distance = -1.0
    sub = np.subtract(p1, p2)
    power = np.power(sub, 2)

    distance = math.sqrt(np.sum(power))

    return distance


def get_sample(data):
    length = len(data)
    sample_size = int(length * 0.01)
    random_nums = set()
    sample_data = []

    for i in range(sample_size):
        random_index = random.randint(0, length - 1)
        while random_index in random_nums:
            random_index = random.randint(0, length - 1)
        random_nums.add(random_index)
        sample_data.append(data[random_index])

    return np.array(sample_data)


def get_new_centroid_addition(centroid, point):
    point = np.array(list(point))
    new_centroid = centroid + point
    return new_centroid


def get_new_centroid_avg(centroid, cluster_point_count):
    new_centroid = np.true_divide(centroid, cluster_point_count)
    return new_centroid


def get_diff_in_j(J, jPrev):
    diff_in_j = np.array(J) - np.array(jPrev)
    return diff_in_j


def compare_jvalue(diff_in_j, j_comparison):
    is_less_than = False

    diff_in_j = np.sum(diff_in_j)
    j_comparison = np.sum(j_comparison)

    if diff_in_j <= j_comparison:
        is_less_than = True
    return is_less_than


def kmeans(data, dimension, k):
    # centroids: [(centroid0 fearures); (centroid1 features); ... ..]
    centroids = initialize_centroids_simple(data, dimension, k)

    # cluster_affiliation: [((point1index  features),clusterindex); ((point2index features), clusterindex)... ]
    cluster_affiliation = [[tuple(features), None] for features in data]

    flag = 1
    jPrev = None
    count = 0

    while flag:
        for i, point in enumerate(data):
            # initializing min_distance and min_distance_index
            min_distance = float('inf')
            min_distance_index = None

            # find closest centroids for each data points
            for cluster_index, centroid in enumerate(centroids):
                if centroid[0] == None:
                    continue

                distance = get_euclidean_distance(centroid, point[1:])

                if distance < min_distance:
                    min_distance = distance
                    min_distance_index = cluster_index

            # record or update cluster for each data points
            if cluster_affiliation[i][1] != min_distance_index:
                cluster_affiliation[i][1] = min_distance_index

                # recompute centroid
        centroids = np.empty((k, dimension))

        # cluster point count will have k number of elements
        cluster_point_count = [0 for _ in range(k)]

        # TO DO
        # write your code to count each cluster pointcount and store them in clutser_point_count structure
        # counting cluster_point_count
        for i in range(k):
            for affiliation in cluster_affiliation:
                if affiliation[1] == i:
                    cluster_point_count[i] += 1

        # recompute new centroids using the count
        for affiliation in cluster_affiliation:
            for cluster_point_index, cluster_point in enumerate(cluster_point_count):
                if affiliation[1] == cluster_point_index:
                    centroids[cluster_point_index] = get_new_centroid_addition(centroids[cluster_point_index],
                                                                               affiliation[0][1:])

        for i in range(len(centroids)):
            if cluster_point_count[i] != 0:
                centroids[i] = get_new_centroid_avg(centroids[i], cluster_point_count[i])

        # TO DO
        # Terminate the while loop based on termination criteria. Write your code to turn flag = false
        all_abs_val = []
        for index, point in enumerate(cluster_affiliation):
            val = np.subtract(list(point[0][1:]), centroids[point[1]])
            abs_val = [np.linalg.norm(elem) ** 2 for elem in val]
            all_abs_val.append(abs_val)

        sum_of_all_abs_val = [sum(x) for x in zip(*all_abs_val)]

        J = [1 / len(data) * s for s in sum_of_all_abs_val]

        if jPrev != None:
            J = [abs(j) for j in J]
            jPrev = [abs(j_prev) for j_prev in jPrev]
            j_comparison = [j * (10 ** -5) for j in J]
            diff_in_j = get_diff_in_j(J, jPrev)
            if compare_jvalue(diff_in_j, j_comparison):
                flag = False
        jPrev = J

        count += 1
    return (centroids, cluster_affiliation)


def gather_clusters_info(centroids, cluster_affiliation):
    clusters_temp = [[tuple(centroid), []] for centroid in centroids]

    for a in cluster_affiliation:
        features, cluster_index = a[0], a[1]
        clusters_temp[cluster_index][1].append(features)

    clusters = []
    for cluster in clusters_temp:
        if cluster[0][0] != None:
            clusters.append(cluster)
    return clusters


def delete_redundant_cluster(clusters, final_count):
    index_count = []

    for index, cluster in enumerate(clusters):
        index_count.append((len(cluster[1]), index))

    # sort with cluster size
    index_count.sort(key=lambda x: x[0])

    # removing the smallest clusters e.g keeping the last 'K' number of clusters who are also largest
    for i in range(len(clusters) - final_count):
        clusters.pop(0)

    return clusters


def initialize_stat(clusters, dimension, cluster_min_size):
    stats = []
    set_point_index = []
    remaining_points = []

    # clusters: [(centroid1, [(point1 features), (point2 features), ...]), (centroid2, ...)]
    # centroid: (centroid features)
    # points: [(point1 51 features); (point2 51 features) ... ... ..] points in the cluster
    for centroid, points in clusters:
        if len(points) >= cluster_min_size:
            # stat: [0, [tuple 50 features]. [tuple 50 features]]
            stat = [0, np.array([0 for _ in range(dimension)], dtype=float),
                    np.array([0 for _ in range(dimension)], dtype=float)]
            point_index = set()

            for point in points:
                point_index.add(point[0])

                stat[0] += 1

                point = np.array(point[1:])

                stat[1] += point
                stat[2] += point ** 2

            # stats: [(numberofpoints in cluster0, SUM(tuple 50 features), SUMSQ (tuple 50 features)); (numberofpoints in cluster1, SUM(tuple 50 features), SUMSQ (tuple 50 features));]
            stats.append(stat)

            # set_point_index: [{point indeices in cluster 0}, {point indices in cluster 1}]
            set_point_index.append(point_index)
        else:
            remaining_points.extend(points)

    return stats, set_point_index, remaining_points


def get_centroids_sd(stat, dimension):
    centroids = []
    cluster_sd = []

    for N, SUM, SUMSQ in stat:
        centroid = np.empty((0, dimension))
        sd = np.empty((0, dimension))

        centroid = SUM / N
        variance = SUMSQ / N - (SUM / N) ** 2
        sd = np.power(variance, 0.5)

        centroids.append(centroid)
        cluster_sd.append(sd)

    return centroids, cluster_sd


def get_mahalanobis_distance(p1, p2, p1_with_index, p2_with_index, sd, dimension):
    # p1 = point
    # p2 = centroid

    if p1_with_index:
        p1 = p1[1:]

    if p2_with_index:
        p2 = p2[1:]

    p = np.true_divide(p1 - p2, sd)
    p = p ** 2
    p = np.sum(p) ** 0.5

    return p


def update_stat(data, stat, set_point_index, dimension, threshold, first_load):
    # for first_load = True
    # data = data file
    # stat = ds
    # set_point_index = ds_point_index

    # for first_load = False
    # data = remaining_points
    # stat = cs
    # set_point_index = cs_point_index

    # set_point_index: [{point indeices in cluster 0}, {point indices in cluster 1}]

    # centroids: [(average  50 tuple); (average  50 tuple); ... ... ]
    # cluster_sd = [(cluster0 sd 50 tuple); (cluster1 sd 50 tuple); ... ... ]
    centroids, cluster_sd = get_centroids_sd(stat, dimension)

    remaining_points = []

    for point in data:
        point = tuple(point)

        if first_load:
            point_exist = False

            for point_index in set_point_index:
                # check whether the point from data already has cluster assignments?
                if point[0] in point_index:  # search the point in the list point_index for corresponding cluster
                    point_exist = True
                    break

            if point_exist:
                continue

        min_mahalanobis_distance = float('inf')
        centroids_min_index = 0

        for index, centroid in enumerate(centroids):
            mahalanobis_distance = get_mahalanobis_distance(point, centroid, True, False, cluster_sd[index], dimension)

            if mahalanobis_distance < min_mahalanobis_distance:
                min_mahalanobis_distance = mahalanobis_distance
                centroids_min_index = index

        if min_mahalanobis_distance < threshold:
            set_point_index[centroids_min_index].add(point[0])  # adding the new point index
            stat[centroids_min_index][0] += 1  # increase point count
            stat[centroids_min_index][1] += np.array(point[1:])
            stat[centroids_min_index][2] += np.array(point[1:]) ** 2
        else:
            remaining_points.append(point)

    return (stat, set_point_index, remaining_points)


def merge_clusters(stat1, point_index1, stat2, point_index2, i, j, dimension, combined_cs):
    s1 = []
    s2 = []

    for n in range(3):
        s1.append(stat1[i][n])
        s2.append(stat2[j][n])
    combined_stat = [s1[0] + s2[0]]

    v1 = []
    v2 = []

    for d in range(dimension):
        v1.append(s1[1][d] + s2[1][d])
        v2.append(s1[2][d] + s2[2][d])
    combined_stat.append(v1)
    combined_stat.append(v2)

    temp = point_index1[i]

    for point in point_index2[j]:
        temp.add(point)

    if combined_cs:
        del stat2[max(i, j)]
        del stat2[min(i, j)]
        del point_index2[max(i, j)]
        del point_index2[min(i, j)]
    else:
        del stat1[i]
        del stat2[j]
        del point_index1[i]
        del point_index2[j]

    stat2.append(combined_stat)
    point_index2.append(temp)

    return (stat1, point_index1, stat2, point_index2)


def check_merge_clusters(stat1, point_index1, stat2, point_index2, dimension, combined_cs, threshold):
    # stat1 = cs
    # point_index1 = cs_point_index
    # stat2 = ds
    # point_index2 = ds_point_index

    while True:
        merged = False

        if not combined_cs:
            if stat1:
                centroids1, cluster_sd1 = get_centroids_sd(stat1, dimension)
                centroids2, cluster_sd2 = get_centroids_sd(stat2, dimension)

                for i in range(len(stat1)):
                    if merged:
                        break

                    for j in range(len(stat2)):
                        mahalanobis_distance = get_mahalanobis_distance(centroids1[i], centroids2[j], False, False,
                                                                        cluster_sd2[j], dimension)

                        if mahalanobis_distance < threshold:
                            stat1, point_index1, stat2, point_index2 = merge_clusters(stat1, point_index1, stat2,
                                                                                      point_index2, i, j, dimension,
                                                                                      False)
                            merged = True
                            break

            if not merged:
                return (stat1, point_index1, stat2, point_index2)

        else:
            cs = stat2
            cs_length = len(cs)

            for i in range(cs_length):
                cs[i][1] = np.array(cs[i][1])
                cs[i][2] = np.array(cs[i][2])

            if cs_length > 1:
                centroids, cluster_sd = get_centroids_sd(cs, dimension)

                for i in range(cs_length - 1):
                    if merged:
                        break

                    for j in range(i + 1, cs_length):
                        mahalanobis_distance = get_mahalanobis_distance(centroids[i], centroids[j], False, False,
                                                                        cluster_sd[j], dimension)

                        if mahalanobis_distance < threshold:
                            stat1, point_index1, stat2, point_index2 = merge_clusters(cs, point_index2, cs,
                                                                                      point_index2, i, j, dimension,
                                                                                      True)
                            merged = True
                            break

            if not merged:
                return (stat2, point_index2)


def main():
    start = time.time()

    inputpath = 'E:\IUB-Courses\Autumn 2021\Data Mining & Warehouse\Assignment3\data'  # sys.argv[1]
    K = 4  # int(sys.argv[2])
    output1 = 'out1'  # sys.argv[3]
    output2 = 'out2'  # sys.argv[4]

    data_num = 0
    data = load(inputpath + '/data' + str(data_num) + '.txt')
    dimension = len(data[0]) - 1

    threshold = 4 * math.sqrt(dimension)
    sample_data = get_sample(data)

    # first time run kmeans clustering with K*3 clusters
    # centroids: [(centroid1 fearues); (centroid2 features)]
    # cluster_affiliation: [((point1 features with point index),group index); ((point2 features with point index),group index)... ]
    centroids, cluster_affiliation = kmeans(sample_data, dimension, K * 3)

    # if not enough clusters, run kmeans again
    while True:
        centroid_count = 0
        for centroid in centroids:
            if centroid[0] != None:
                centroid_count += 1
        if centroid_count < K:
            centroids, cluster_affiliation = kmeans(sample_data, dimension,
                                                    K * 3)  # randomize and restart kmeans until get K or more clusters
        else:
            break

    # clusters: [(centroid1, [(point1 features), (point2 features), ...]), (centroid2, ...)]
    clusters = gather_clusters_info(centroids, cluster_affiliation)

    # delete redundant clutser
    clusters = delete_redundant_cluster(clusters, K)
    # print('clusters: ', clusters)

    # ds: [(number of points N in cluster 0, [SUM], [SUMSQ]), (number of points N in cluster 1, [SUM], [SUMSQ]), ...] 
    # ds_point_index:[{cluster0 point indices}, {cluster1 point indices}, .....]

    ds, ds_point_index, temp = initialize_stat(clusters, dimension, 1)  # based on preliminary clustering
    # print('ds: ', ds)
    # print('ds_point_index: ', ds_point_index)
    # print('temp: ', temp)

    first_load = True
    cs = []
    rs = []
    cs_point_index = []
    inter_results = []
    loop = True

    # loop begins for different chunks
    while loop:
        try:
            # just to check this is the last chunk of data
            data = load(inputpath + '/data' + str(data_num) + '.txt')
        except:
            cs, cs_point_index, ds, ds_point_index = check_merge_clusters(cs, cs_point_index, ds, ds_point_index,
                                                                          dimension, False, threshold)
            break

        # assign points to ds
        ds, ds_point_index, remaining_points = update_stat(data, ds, ds_point_index, dimension, threshold,
                                                           first_load)  # assign new points to ds
        # print('ds: ', ds)
        # print('ds_point_index: ', ds_point_index)
        # print('temp: ', remaining_points)

        # remaining_points: [{point tuple}, {point tuple}, .....]
        if first_load:
            first_load = False

        if remaining_points:
            if cs:
                # merge cs if needed
                # cs: [(number of points N in CScluster 0, [SUM], [SUMSQ]), (number of points N in CS cluster 1, [SUM], [SUMSQ]), ...] 
                # cs_point_index:[{CScluster0 point indices}, {CScluster1 point indices}, .....]

                cs, cs_point_index = check_merge_clusters(cs, cs_point_index, cs, cs_point_index, dimension, True,
                                                          threshold)
                # print('cs: ', cs)
                # print('cs_point_index: ', cs_point_index)

                # assign points to cs
                cs, cs_point_index, remaining_points = update_stat(remaining_points, cs, cs_point_index, dimension,
                                                                   threshold, False)
                # print('cs: ', cs)
                # print('cs_point_index: ', cs_point_index)
                # print('remaining_points: ', remaining_points)

            # print('remaining_points: ', remaining_points)
            # print('rs: ', rs)
            # print(type(remaining_points))
            # print(type(rs))
            _remaining_points = remaining_points.copy()
            _rs = rs.copy()

            for i in range(len(remaining_points)):
                _remaining_points[i] = list(remaining_points[i])

            for i in range(len(rs)):
                _rs[i] = list(rs[i])
                _remaining_points.append(_rs[i])

            centroids, cluster_affiliation = kmeans(_remaining_points, dimension, 3 * K)
            # print('centroids: ', centroids)
            # print('cluster_affiliation: ', cluster_affiliation)

            clusters = gather_clusters_info(centroids, cluster_affiliation)
            # print('clusters: ', clusters)

            cs_temp, cs_point_index_temp, rs = initialize_stat(clusters, dimension, 2)
            print('cs_temp: ', cs_temp)
            print('cs_point_index_temp: ', cs_point_index_temp)
            print('rs: ', rs)

            cs.extend(cs_temp)

            cs_point_index.extend(cs_point_index_temp)

        data_num += 1
        ds_point_count = 0
        cs_point_count = 0
        inter_results.append((data_num, len(ds), sum([len(points) for points in ds_point_index]), len(cs),
                              sum([len(points) for points in cs_point_index]), len(rs)))

        print(inter_results)

        if data_num == 5:
            loop = False

    results = {}
    for index, points in enumerate(ds_point_index):
        for point in points:
            results[str(point)] = index

    for points in cs_point_index:
        for point in points:
            results[str(point)] = -1

    for point in rs:
        results[str(point[0])] = -1

    fh = open(output1, 'w')

    json.dump(results, fh)

    fh.close()

    fh = open(output2, 'w')
    fh.write(
        'round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_compression,nof_point_retained')

    for line in inter_results:
        fh.write('\n')
        fh.write(str(line).strip('()'))

    fh.close()
    # loop = False
    '''
    '''
    print('Duration: %s' % (time.time() - start))


if __name__ == "__main__":
    main()
