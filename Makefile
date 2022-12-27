DOCKER_COMPOSE = cd docker/ && docker compose -f docker-compose.yml

.SILENT:
.PHONY: build
build:
	$(DOCKER_COMPOSE) build

.SILENT:
.PHONY: shell
shell:
	docker container exec -it snakamura_test bash


.PHONY: prune
prune:
	docker system prune

.PHONY: delete_DS
delete_DS:
	find . -name ".DS_Store" | xargs rm