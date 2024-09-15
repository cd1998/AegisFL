from seal import *
import numpy as np
import time

def compute_sum():
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 8192
    model_number = 101770
    scale = 2 ** 40
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 60]))
    parms.set_encoding_method(encoding_method.pm)
    context = SEALContext(parms)
    # context.get_context_data(context.first_parms_id()).parms().set_encoding_method(encoding_method.pm1)
    keygen = KeyGenerator(context)
    secret_key = keygen.secret_key()
    public_key = keygen.create_public_key()
    relin_keys = keygen.create_relin_keys()

    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)
    encoder = CKKSEncoder(context)

    data1 = np.random.rand(model_number)
    data2 = np.ones(model_number)
    sum1 = np.sum(data1)
    print('sum1:', sum1)
    mod = np.mod(model_number, poly_modulus_degree)
    data1 = np.concatenate([data1, np.zeros(poly_modulus_degree - mod)])
    data2 = np.concatenate([data2, np.zeros(poly_modulus_degree - mod)])

    num_batches = len(data1) // poly_modulus_degree
    encrypt_result_1 = []
    encode_result_2 = []
    for i in range(num_batches):
        start_idx = i * poly_modulus_degree
        end_idx = (i + 1) * poly_modulus_degree
        batch_1 = data1[start_idx:end_idx]
        batch_2 = data2[start_idx:end_idx]
        reversed_part = -batch_2[-1:0:-1]
        result_array = np.concatenate(([batch_2[0]], reversed_part))
        batch_2 = result_array

        plain1 = encoder.encode(batch_1, scale)
        plain2 = encoder.encode(batch_2, scale)
        cipher1 = encryptor.encrypt(plain1)

        encrypt_result_1.append(cipher1)
        encode_result_2.append(plain2)

    inner_pm = []
    start_time = time.time()
    for i in range(num_batches):
        cip_mul = evaluator.multiply_plain(encrypt_result_1[i], encode_result_2[i])
        inner_pm.append(cip_mul)
    end_time = time.time()
    print('e-s', end_time - start_time)
    decrypt = []
    for i in range(num_batches):
        dec = decryptor.decrypt(inner_pm[i])
        pla = encoder.decode(dec)
        decrypt.append(pla)
    sum2 = sum(decrypt)
    print('sum2', sum2)
    return end_time-start_time

def compute_inner():
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 8192
    model_number = 101770
    scale = 2 ** 40
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 60]))
    parms.set_encoding_method(encoding_method.pm)
    context = SEALContext(parms)
    # context.get_context_data(context.first_parms_id()).parms().set_encoding_method(encoding_method.pm1)
    keygen = KeyGenerator(context)
    secret_key = keygen.secret_key()
    public_key = keygen.create_public_key()
    relin_keys = keygen.create_relin_keys()

    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)
    encoder = CKKSEncoder(context)

    data1 = np.random.rand(model_number)
    data2 = np.random.rand(model_number)
    inner = np.dot(data1, data2)
    #L2 = np.linalg.norm(data1)
    print('inner:', inner)
    mod = np.mod(model_number, poly_modulus_degree)
    data1 = np.concatenate([data1, np.zeros(poly_modulus_degree - mod)])
    data2 = np.concatenate([data2, np.zeros(poly_modulus_degree - mod)])

    num_batches = len(data1) // poly_modulus_degree
    encrypt_result_1 = []
    encrypt_result_2 = []
    for i in range(num_batches):
        start_idx = i * poly_modulus_degree
        end_idx = (i + 1) * poly_modulus_degree
        batch_1 = data1[start_idx:end_idx]
        batch_2 = data2[start_idx:end_idx]
        reversed_part = -batch_2[-1:0:-1]
        result_array = np.concatenate(([batch_2[0]], reversed_part))
        batch_2 = result_array

        plain1 = encoder.encode(batch_1, scale)
        plain2 = encoder.encode(batch_2, scale)
        cipher1 = encryptor.encrypt(plain1)
        cipher2 = encryptor.encrypt(plain2)

        encrypt_result_1.append(cipher1)
        encrypt_result_2.append(cipher2)

    inner_pm = []
    start_time = time.time()
    for i in range(num_batches):
        cip_mul = evaluator.multiply(encrypt_result_1[i], encrypt_result_2[i])
        evaluator.relinearize_inplace(cip_mul, relin_keys)
        inner_pm.append(cip_mul)
    end_time = time.time()
    print('e-s', end_time - start_time)
    decrypt = []
    for i in range(num_batches):
        dec = decryptor.decrypt(inner_pm[i])
        pla = encoder.decode(dec)
        decrypt.append(pla)
    inner_sum = sum(decrypt)
    print('inner_sum', inner_sum)
    return end_time-start_time


if __name__ == "__main__":
    total_time = []
    for i in range(10):
        need_time = compute_inner()
        total_time.append(need_time)
    mean_time = sum(total_time)/len(total_time)
    print('mean_time', mean_time)