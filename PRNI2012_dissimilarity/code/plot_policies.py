import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    #prototype_policies = ['draw', 'subset', 'cover', 'kmeans', 'kcenter', 'sff', 'kmeans++']
    prototype_policies = ['subset', 'kcenter', 'sff']
    marker_policy = {'draw':'ko-', 'subset':'k+-', 'cover':'k^-', 'kmeans':'kx-', 'kcenter':'k*-', 'sff':'kD-', 'kmeans++':'k-'}
    label_policy = {'draw':'random($P_X$)', 'subset':'random($S$)', 'cover':'constrained rand($S$)', 'kmeans':'clusters', 'kcenter':'k-center', 'sff':'sff', 'kmeans++':'kmeans++'}
    color_policies = ['r', 'g', 'b', 'w', 'r', 'c', 'black']    
    i = -1
    iterations = 50 

    plt.figure()
    for policy in prototype_policies:
        i = i +1
        print policy
        filename = 'score_sample_gen_'+policy+'.npy'        
        #filename = '120229_16h_tracks_score_gen_'+policy+'.npy'
        #filename = '120229_20h_40pro_tracks_score_gen_'+policy+'.npy'        
        #filename = '120301_14h_20pro_tracks_score_gen_'+policy+'.npy'
        #filename = '120301_17h_20pro_100tracks_score_gen_'+policy+'.npy'        
        #filename = '120301_20h_400_10pro_c10_tracks_score_gen_'+policy+'.npy'        
        #filename = '120305_19h_100_05pro_c10_tracks_score_gen_'+policy+'.npy'
        #filename = '120309_15h_20_01pro_c10_100tracks_score_gen_'+policy+'.npy'            
        filename = '120309_15h_300_10pro_c10_alltracks_score_gen_'+policy+'.npy'        
        
        score = np.load(filename)
        num_prototypes = score.shape[0]
        #nums_prototypes = range(num_prototypes)

        #nums_prototypes = range(1,num_prototypes*5,5)
        #nums_prototypes = range(num_prototypes)
        nums_prototypes = range(1,num_prototypes*10,10)
        
        score_mean = score.mean(1)
        score_std = score.std(1)
        std_mean = score_std / np.sqrt(iterations)
        print("Average Score corr: %s" % score_mean)
        print("Std Score corr: %s" % score_std)
        plt.plot(nums_prototypes, score_mean, marker_policy[policy], label=label_policy[policy],color = color_policies[i])
        #plt.plot(nums_prototypes, score_mean+std_mean, marker_policy[policy], label=label_policy[policy],color = color_policies[i])
        #plt.plot(nums_prototypes, score_mean-std_mean, marker_policy[policy], label=label_policy[policy],color = color_policies[i])        
        print("Std mean corr: %s" % std_mean)


#    i=2
#    filename = '120301_18h_20pro_100tracks_score_gen_sff_c80.npy'
#    score = np.load(filename)
#    num_prototypes = score.shape[0]
#    nums_prototypes = range(num_prototypes)
#    score_mean = score.mean(1)
#    score_std = score.std(1)
#    std_mean = score_std / np.sqrt(iterations)
#    print("Average Score corr: %s" % score_mean)
#    print("Std Score corr: %s" % score_std)
#    plt.plot(nums_prototypes, score_mean, marker_policy[policy], label=label_policy[policy],color = color_policies[i])
#    plt.plot(nums_prototypes, score_mean+std_mean, marker_policy[policy], label=label_policy[policy],color = color_policies[i])
#    plt.plot(nums_prototypes, score_mean-std_mean, marker_policy[policy], label=label_policy[policy],color = color_policies[i])        
#    print("Std mean corr: %s" % std_mean)
      
    #plt.xlim([1, 5*score.shape[0]])
    #plt.xlim([1, score.shape[0]])
    plt.xlim([1, 10*score.shape[0]])
    plt.ylim([0.5, 0.85])
    plt.legend(loc = 'lower right')
    plt.xlabel('number of prototypes ($p$)')
    plt.ylabel('correlation')
    plt.show()
