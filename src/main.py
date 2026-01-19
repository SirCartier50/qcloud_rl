from job import job_generator


def main():
    # Your main code goes here
    a = job_generator()
    job_queue = a.generate_job(10)
    print(len(job_queue))

if __name__ == '__main__':
    main()