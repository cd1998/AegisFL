from seal import *
import numpy as np
import time

def compute_sum():
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 60]))
    scale = 2.0 ** 40
    model_number = 101770

    context = SEALContext(parms)
    ckks_encoder = CKKSEncoder(context)
    slot_count = ckks_encoder.slot_count()
    print('slot_count', slot_count)
    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.create_relin_keys()
    galois_keys = keygen.create_galois_keys()
    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)

    def compute_rotate(cip, i, key):
        result = evaluator.rotate_vector(cip, 2 ** i, key)
        return result

    data1 = np.random.rand(model_number)
    sum1 = np.sum(data1)
    print('sum1:', sum1)

    num_batches = len(data1) // slot_count + 1
    encrypt_result_1 = []

    for i in range(num_batches):
        start_idx = i * slot_count
        end_idx = (i + 1) * slot_count
        batch_1 = data1[start_idx:end_idx]

        plain1 = ckks_encoder.encode(batch_1, scale)
        cipher1 = encryptor.encrypt(plain1)

        encrypt_result_1.append(cipher1)

    sum_ckks = []
    start_time = time.time()
    for i in range(num_batches):
        for j in range(int(np.log2(slot_count))):
            c = compute_rotate(encrypt_result_1[i], j, galois_keys)
            evaluator.add_inplace(encrypt_result_1[i], c)
        sum_ckks.append(encrypt_result_1[i])
    end_time = time.time()
    print('total_time:', end_time - start_time)

    decrypt = []
    for i in range(num_batches):
        dec = decryptor.decrypt(sum_ckks[i])
        pla = ckks_encoder.decode(dec)
        decrypt.append(pla)
    sum2 = sum(decrypt)
    print('sum2', sum2)

    return end_time-start_time


def compute_inner():
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 60]))
    scale = 2.0 ** 40
    model_number = 101770

    context = SEALContext(parms)
    ckks_encoder = CKKSEncoder(context)
    slot_count = ckks_encoder.slot_count()
    print('slot_count', slot_count)
    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.create_relin_keys()
    galois_keys = keygen.create_galois_keys()
    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)

    def compute_rotate(cip, i, key):
        result = evaluator.rotate_vector(cip, 2 ** i, key)
        return result

    data1 = np.random.rand(model_number)
    data2 = np.random.rand(model_number)
    #data2 = data1
    inner = np.dot(data1, data2)
    #L2 = np.linalg.norm(data1)
    print('inner:', inner)

    num_batches = len(data1) // slot_count + 1

    encrypt_result_1 = []
    encrypt_result_2 = []
    for i in range(num_batches):
        start_idx = i * slot_count
        end_idx = (i + 1) * slot_count
        batch_1 = data1[start_idx:end_idx]
        batch_2 = data2[start_idx:end_idx]

        plain1 = ckks_encoder.encode(batch_1, scale)
        plain2 = ckks_encoder.encode(batch_2, scale)
        cipher1 = encryptor.encrypt(plain1)
        cipher2 = encryptor.encrypt(plain2)

        encrypt_result_1.append(cipher1)
        encrypt_result_2.append(cipher2)

    inner_ckks = []
    start_time = time.time()
    for i in range(num_batches):
        cip_mul = evaluator.multiply(encrypt_result_1[i], encrypt_result_2[i])
        evaluator.relinearize_inplace(cip_mul, relin_keys)
        for i in range(int(np.log2(slot_count))):
            c = compute_rotate(cip_mul, i, galois_keys)
            evaluator.add_inplace(cip_mul, c)
        inner_ckks.append(cip_mul)
    end_time = time.time()
    print('total_time:', end_time - start_time)

    decrypt = []
    for i in range(num_batches):
        dec = decryptor.decrypt(inner_ckks[i])
        pla = ckks_encoder.decode(dec)
        decrypt.append(pla)
    inner_sum = sum(decrypt)
    print('sum', inner_sum)

    return end_time-start_time

if __name__ == "__main__":
    total_time = []
    for i in range(10):
        need_time = compute_inner()
        total_time.append(need_time)
    mean_time = sum(total_time)/len(total_time)
    print('mean_time', mean_time)




