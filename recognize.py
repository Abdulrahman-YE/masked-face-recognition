from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


# comparing against the base, could be a prototype (k * 512) or all embedding from database (k * 3 * 512)
def compare_embed(base, embed, metrics='euclidean'):
    """
        المقارنة مع قاعدة البيانات 
    """
    if metrics == 'euclidean':
        return euclidean_distances(base, embed)
    if metrics == 'cosine':
        return cosine_distances(base, embed)

# check comparison result against prototype with threshold
def prototype_compare_result(arr, threshold):
    """
        تحديد ناتج المقارنة مع الملامح الاخرى و مقارنتها مع العتبه المحدده
        المخلات : 
        المدخلات :
        - arr : مصفوفة الملامح
        - threshold : العتبة التي لا يمكن ان تقل نسبة التشابه عنها
    """
    if arr.min() > threshold:
        return None,None
    else:
        return arr.argmin(), arr.min() 


def recognize(db, embed, mode='prototype'):
    """
        تتعرف على هوية الشخص عن طريق مقارنت ملامحه بملامح الاشخاض في قاعدة البيانات.
        المدخلات
         - db : مصفوفة تحتوي على ملامح الاشخاص في قاعدة البيانات
         - mode : طريقة المقارنه المتبعة    
         المخرج:
         -  الخانة التي فيها اكثر نسبة تشابه
         - نسبة التشابة
    """
    mode = mode # prototype or each
    metrics = 'cosine' # euclidean, cosine, cond_prob
    threshold = 0.55

    return prototype_compare_result(compare_embed(db, embed.reshape(1,512), metrics=metrics), threshold=threshold)
